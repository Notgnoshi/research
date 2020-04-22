import glob
import json
import logging
import pathlib
import pprint
import re
import shutil
import sys
from typing import List

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from ..data import get_data_dir, get_df
from ..nlp import preprocess
from .base import LanguageModel


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
}


class HaikuDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, evaluation=False, eval_ratio=0.1, block_size=512,
    ):
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from haiku dataset")

        df = get_df()
        for idx, row in df.iterrows():
            haiku = "^ " + row["haiku"] + " $"
            df.at[idx, "haiku"] = haiku

        logger.info("Shuffling and splitting dataset.")
        df = df.sample(frac=1).reset_index(drop=True)
        split = int(len(df) * eval_ratio)
        if evaluation:
            self.examples = tokenizer.batch_encode_plus(
                df["haiku"][:split], add_special_tokens=True, max_length=block_size
            )["input_ids"]
        else:
            self.examples = tokenizer.batch_encode_plus(
                df["haiku"][split:], add_special_tokens=True, max_length=block_size
            )["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class TransformerModel(LanguageModel):
    """A transformer-powered causal language model for text generation."""

    def __init__(self, config: dict, quiet=True):
        super().__init__(config, quiet)

        self.model_type = config["model_type"]
        self.evaluate_during_training = config["evaluate_during_training"]
        self.evaluation_proportion = config["evaluation_proportion"]
        self.batch_size = config["batch_size"]
        self.gradient_accumulation_steps = config["gradient_accumulation_steps"]
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.adam_epsilon = config["adam_epsilon"]
        self.max_gradient_norm = config["max_gradient_norm"]
        self.num_train_epochs = config["num_train_epochs"]
        self.max_steps = config["max_steps"]
        self.warmup_steps = config["warmup_steps"]
        self.logging_steps = config["logging_steps"]
        self.checkpoint_steps = config["checkpoint_steps"]
        self.resume_training_from = config["resume_training_from"]
        self.max_checkpoints = config["max_checkpoints"]

        # Disable CUDA because the GPT2 model doesn't fit on my GTX 1080, and it really inflates the size of the container.
        self.cuda = False

        self.temperature = config["temperature"]
        self.repetition_penalty = config["repetition_penalty"]
        self.top_k = config["k"]
        self.top_p = config["p"]

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_type]

        cache = get_data_dir() / "cache"

        model_name = self.model_type
        # Restore model from checkpoint.
        if self.resume_training_from is not None:
            checkpoint = self.output_directory / self.resume_training_from
            if not checkpoint.exists():
                logger.error("Could not resume from nonexistent checkpoint %s", checkpoint)
                sys.exit(1)

            model_name = str(checkpoint)

        self.lm_config: PretrainedConfig = config_class.from_pretrained(model_name, cache_dir=cache)
        self.tokenizer: PreTrainedTokenizer = tokenizer_class.from_pretrained(
            model_name, cache_dir=cache
        )

        self.train_dataset = HaikuDataset(self.tokenizer, block_size=self.tokenizer.max_len)
        self.eval_dataset = HaikuDataset(
            self.tokenizer,
            evaluation=True,
            eval_ratio=self.evaluation_proportion,
            block_size=self.tokenizer.max_len,
        )

        self.model: PreTrainedModel = model_class.from_pretrained(
            model_name, config=self.lm_config, cache_dir=cache
        )

        self.model.to(self.device)

    def _serialize(self, directory: pathlib.Path):
        """Save the trained model to disk."""
        logger.info("Saving trained model to %s", directory)
        try:
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            model_to_save.save_pretrained(str(directory))
            self.tokenizer.save_pretrained(str(directory))
            torch.save(self.config, directory / "self.config.bin")
        except:
            return False
        return True

    def _deserialize(self, directory: pathlib.Path):
        """Load the trained model from disk."""
        logger.info("Loading fine-tuned model and vocabulary from %s", directory)
        _, model_class, tokenizer_class = MODEL_CLASSES[self.model_type]
        try:
            self.model = model_class.from_pretrained(str(directory))
            self.tokenizer = tokenizer_class.from_pretrained(str(directory))
            self.model.to(self.device)
        except:
            return False
        return True

    def train(self):
        """Use self.config parameters to train the model."""

        def collate(example: List[torch.Tensor]):
            if self.tokenizer._pad_token is None:
                return pad_sequence(example, batch_first=True)
            return pad_sequence(
                example, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

        sampler = RandomSampler(self.train_dataset)
        dataloader = DataLoader(
            self.train_dataset, sampler=sampler, batch_size=self.batch_size, collate_fn=collate
        )

        if self.max_steps is not None:
            t_total = self.max_steps
            self.num_train_epochs = (
                self.max_steps // (len(dataloader) // self.gradient_accumulation_steps) + 1
            )
        else:
            t_total = len(dataloader) // self.gradient_accumulation_steps * self.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_params = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_params, lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=t_total
        )

        logger.info("===== Running training =====")
        logger.info("  Examples:                    %d", len(self.train_dataset))
        logger.info("  Epochs:                      %d", self.num_train_epochs)
        logger.info("  Batch size:                  %d", self.batch_size)
        logger.info("  Gradient accumulation steps: %d", self.gradient_accumulation_steps)
        logger.info("  Total optimization steps:    %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_epoch = 0

        if self.resume_training_from is not None:
            checkpoint = self.output_directory / self.resume_training_from
            checkpoint = checkpoint.resolve()
            if not checkpoint.exists():
                logger.error("Could not resume training from nonexistent checkpoint %s", checkpoint)
                sys.exit(1)

            global_step = int(checkpoint.stem.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(dataloader) // self.gradient_accumulation_steps)
            steps_trained_in_epoch = global_step % (
                len(dataloader) // self.gradient_accumulation_steps
            )

            logger.info("  Resuming from checkpoint.")
            logger.info("  Resuming from epoch: %d", epochs_trained)
            logger.info("  Resuming from step:  %d", global_step)
            logger.info(
                "  Will skip the first %d steps of the current epoch", steps_trained_in_epoch
            )

            optimizer.load_state_dict(torch.load(str(checkpoint / "optimizer.bin")))
            scheduler.load_state_dict(torch.load(str(checkpoint / "scheduler.bin")))

        tr_loss, logging_loss = 0.0, 0.0
        model_to_resize = self.model.module if hasattr(self.model, "module") else self.model
        model_to_resize.resize_token_embeddings(len(self.tokenizer))
        self.model.zero_grad()

        train_iterator = trange(epochs_trained, self.num_train_epochs, desc="Epoch", disable=False)
        for _ in train_iterator:
            epoch_iterator = tqdm(dataloader, desc="Iteration", disable=False)
            for step, batch in enumerate(epoch_iterator):
                # Skip ahead to where we left off.
                if steps_trained_in_epoch > 0:
                    steps_trained_in_epoch -= 1
                    continue

                inputs, labels = (batch, batch)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.model.train()
                outputs = self.model(inputs, labels=labels)
                loss = outputs[0]

                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    # Periodically evaluate the model during training.
                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        if self.evaluate_during_training:
                            logger.info("---- Evaluation at step: %d ----", global_step)
                            logger.info("  lr:   %f", scheduler.get_lr()[0])
                            logger.info("  loss: %f", (tr_loss - logging_loss) / self.logging_steps)
                            self.evaluate()
                        logging_loss = tr_loss

                    # Make periodic checkpoints that can be restored from
                    if self.checkpoint_steps > 0 and global_step % self.checkpoint_steps == 0:
                        checkpoint: pathlib.Path = self.output_directory / f"checkpoint-{global_step}"
                        checkpoint.mkdir(exist_ok=True)
                        logger.info("Saving checkpoint to %s", checkpoint)
                        self.serialize(checkpoint)
                        torch.save(optimizer.state_dict(), str(checkpoint / "optimizer.bin"))
                        torch.save(scheduler.state_dict(), str(checkpoint / "scheduler.bin"))

                        self.rotate_checkpoints()

                if self.max_steps is not None and global_step > self.max_steps:
                    epoch_iterator.close()
                    break
            if self.max_steps is not None and global_step > self.max_steps:
                train_iterator.close()

        logger.info("global_step:  %d", global_step)
        logger.info("average loss: %f", tr_loss / global_step)

        # We need to load the fine-tuned tokenizer, so we serialize, then deserialize.
        self.serialize()
        torch.save(optimizer.state_dict(), str(checkpoint / "optimizer.bin"))
        torch.save(scheduler.state_dict(), str(checkpoint / "scheduler.bin"))
        self.deserialize()

        logger.info("Final evaluation.")
        self.evaluate()

    def sorted_checkpoints(self):
        checkpoints = glob.glob(self.output_directory / "checkpoint-*")

        def atoi(s: str):
            return int(s) if s.isdigit() else s

        def keys(s: str):
            return [atoi(c) for c in re.split(r"(\d+)", s)]

        return sorted(checkpoints, key=keys)

    def rotate_checkpoints(self):
        if self.max_checkpoints is None:
            return
        if self.max_checkpoints <= 0:
            return

        checkpoints = self.sorted_checkpoints()
        num_delete = max(0, len(checkpoints) - self.max_checkpoints)
        delete = checkpoints[:num_delete]
        for checkpoint in delete:
            logger.info("Deleting checkpoint %s", checkpoint)
            shutil.rmtree(checkpoint)

    def evaluate(self):
        def collate(examples: List[torch.tensor]):
            if self.tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(
                examples, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

        sampler = SequentialSampler(self.eval_dataset)
        dataloader = DataLoader(
            self.eval_dataset, sampler=sampler, batch_size=self.batch_size, collate_fn=collate
        )
        logger.info("  examples: %d", len(self.eval_dataset))
        logger.info("  batch size: %d", self.batch_size)
        loss = 0.0
        eval_steps = 0

        self.model.eval()

        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = (batch, batch)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs, labels=labels)
                loss += outputs[0].mean().item()
            eval_steps += 1

        loss = loss / eval_steps
        perplexity = torch.exp(torch.tensor(loss)).item()

        results = {
            "loss": loss,
            "perplexity": perplexity,
        }

        logger.info("  %s", pprint.pformat(results))
        with open(self.output_directory / "evaluation_results.json", "w") as fd:
            json.dump(results, fd)

    def generate(self, n: int = None) -> pd.DataFrame:
        """Generate n sequences."""
        logger.info("Generating haiku...")
        n = n or self.number
        # Manually insert the begin-of-haiku tag
        prompt = self.tokenizer.encode(
            "^ " + self.prompt, add_special_tokens=False, return_tensors="pt"
        )
        prompt = prompt.to(self.device)

        output_sequences = self.model.generate(
            input_ids=prompt,
            max_length=self.max_tokens + len(prompt[0]),
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
            num_return_sequences=n,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences) > 2:
            output_sequences.squeeze_()

        generated = []
        # Decode the model output back into regular text.
        for sequence in output_sequences:
            sequence = sequence.tolist()

            # Decode the model output into text.
            text = self.tokenizer.decode(sequence, clean_up_tokenization_spaces=True)
            logger.debug("Model output: %s", text)

            # Remove all text after the $
            text = text[: text.find("$")]
            # Add the prompt to the beginning, and remove any excess text.
            text = (
                self.prompt
                + text[len(self.tokenizer.decode(prompt[0], clean_up_tokenization_spaces=True)) :]
            )
            text = preprocess(text)
            generated.append(text)
            logger.info("Generated: %s", text)

        columns = {
            "model": [self.name] * n,
            "type": [f"{self.type}-{self.model_type}"] * n,
            "seed": [self.seed] * n,
            "prompt": [self.prompt] * n,
            "haiku": generated,
        }
        return pd.DataFrame(columns)

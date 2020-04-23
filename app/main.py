import logging
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Path, Query, status
from fastapi.responses import PlainTextResponse, RedirectResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedModel, PreTrainedTokenizer

from haikulib.data import get_data_dir, get_random_prompt
from haikulib.nlp import preprocess

app = FastAPI()
logger = logging.getLogger(__name__)


def load_model() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    # TODO: Set the model path more intelligently.
    model_path = get_data_dir() / "models" / "gpt2"
    model: PreTrainedModel = GPT2LMHeadModel.from_pretrained(str(model_path))
    tokenizer: PreTrainedTokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
    return model, tokenizer


# FastAPI uses both threads, processes, and async event loops to serve the API.
# PyTorch models are thread-safe, but I don't know about transformers models.
# In any case, I don't know what to do other than load it here (a per-process global)
# and hope there's nothing too dangerous about concurrent uses.
# If it's not thread-safe, I might need to add some synchronization?
# Or FastAPI allows configuring the number of workers. I should set #workers = some small number.
model, tokenizer = load_model()


def generate(
    prompt: str, seed: int, number: int, temperature: float, k: int, p: float, max_tokens: int
) -> pd.DataFrame:
    # TODO: Is this thread-safe? I doubt it...
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    encoded_prompt = tokenizer.encode("^ " + prompt, add_special_tokens=False, return_tensors="pt")
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_tokens + len(encoded_prompt[0]),
        temperature=temperature,
        top_k=k,
        top_p=p,
        do_sample=True,
        num_return_sequences=number,
    )
    if len(output_sequences) > 2:
        output_sequences.squeeze_()

    generated = []
    for sequence in output_sequences:
        sequence = sequence.tolist()
        text = tokenizer.decode(sequence, clean_up_tokenization_spaces=True)
        text = text[: text.find("$")]
        text = (
            prompt
            + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )
        text = preprocess(text)
        generated.append(text)
        logger.info("Generated: %s", text)

    columns = {
        "model": ["fastapi-gpt2"] * number,
        "type": ["transformer-gpt2"] * number,
        "seed": [seed] * number,
        "prompt": [prompt] * number,
        "haiku": generated,
    }
    return pd.DataFrame(columns)


@app.get("/generate")
def generate_haiku(
    prompt: str = Query(
        None,
        description="The prompt to begin the generated haiku with. If not given, a random one will be chosen.",
        max_length=50,
    ),
    seed: int = Query(
        None,
        description="Seed the RNG. If not given, a random seed will be generated, used, and returned in the JSON response for reproducibility.",
        gt=0,
        lt=2 ** 32,
    ),
    number: int = Query(5, description="The number of haiku to generate.", gt=0, le=20),
    temperature: float = Query(
        1.0,
        description="The temperature to use when generating the haiku. Higher temperatures result in more randomness.",
        gt=0,
    ),
    k: int = Query(
        0,
        description="The number of highest probability vocabulary tokens to keep for top-k filtering.",
        ge=0,
    ),
    p: float = Query(
        0.9,
        description="The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling.",
        ge=0,
        le=1,
    ),
    max_tokens: int = Query(
        20, description="The max length of the sequence to be generated.", gt=0,
    ),
):
    """Generate a random haiku based on the given prompt."""
    if prompt is None:
        prompt = get_random_prompt()
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    df = generate(prompt, seed, number, temperature, k, p, max_tokens)
    logger.debug("Saving generated DataFrame to data/generated.csv")
    with open(get_data_dir() / "generated.csv", "a", encoding="utf-8", errors="ignore") as f:
        df.to_csv(f, mode="a", header=(f.tell() == 0), index=False)

    # TODO: This is a fairly expensive request. Profile and see what it would take to optimize.
    # TODO: Return the index for each generated haiku so that they're retrievable from the /generated/{n} links
    return {"seed": seed, "prompt": prompt, "haiku": df["haiku"]}


@app.get("/generated")
async def random_generated_haiku():
    """Get a random generated haiku."""
    # TODO: Dynamically update the upper bound based on the number of generated haiku.
    # Should try to avoid file I/O for this, however.
    n = random.randrange(0, 1200)
    return RedirectResponse(f"/generated/{n}")


# TODO: Find a way to cache the generated haiku in a thread-safe manner.
@app.get("/generated/{n}", response_class=PlainTextResponse)
def generated_haiku(n: int = Path(..., description="The haiku index", ge=0)):
    """Get the nth generated haiku."""
    raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Not implemented.")

    # TODO Dynamically update the upper bound.
    if n > 1200:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="That haiku doesn't exist!")
    return f"{n} spoonfuls / of medication / and loneliness"


@app.get("/data")
async def random_training_set_haiku():
    """Get a random human-written haiku from the training set."""
    # TODO: Pick this value at the API start time.
    n = random.randrange(0, 55000)
    return RedirectResponse(f"/data/{n}")


# TODO: Find a way to keep the training set dataframe in memory in a thread-safe manner.
@app.get("/data/{n}")
def training_set_haiku(n: int = Path(..., description="The haiku index", ge=0)):
    """Get the nth human-written haiku from the training set."""
    raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Not implemented.")

    # TODO: Update this value at API start time.
    if n > 55000:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="That haiku doesn't exist!")
    return f"{n} a love letter / to the butterfly gods / with strategic misspellings"

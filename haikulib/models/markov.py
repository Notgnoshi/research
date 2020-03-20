import logging
import pathlib
import pickle
import random

import nltk
import nltk.lm
import pandas as pd

from .base import LanguageModel

logger = logging.getLogger(__name__)


class MarkovModel(LanguageModel):
    """A Kneser-Ney smoothed Markov language model."""

    def __init__(self, config: dict, quiet=True):
        super().__init__(config, quiet)
        self.order = config["order"]
        self.seed_tokens = config["seed_tokens"]
        self.vocab = nltk.lm.Vocabulary(self.bag)
        self.model = nltk.lm.models.KneserNeyInterpolated(order=self.order, vocabulary=self.vocab)

    def _serialize(self, filename: pathlib.Path):
        """Save the trained model to disk."""
        with open(filename, "wb") as fd:
            pickle.dump(self.model, fd)
        return True

    def _deserialize(self, filename: pathlib.Path):
        """Load the trained model from disk."""
        with open(filename, "rb") as fd:
            self.model = pickle.load(fd)
        return True

    def train(self):
        if not self.quiet:
            logger.info("Training model...")
            logger.info("tokenizing...")
        if self.tokenization == "words":
            tokens = nltk.word_tokenize(self.corpus)
        elif self.tokenization == "characters":
            tokens = list(self.corpus)
        ngrams = nltk.everygrams(tokens, max_len=self.order)
        if not self.quiet:
            logger.info("fitting...")
        self.model.fit([ngrams], vocabulary_text=self.vocab)
        if not self.quiet:
            logger.info("Trained model.")

    def _generate(self, seed) -> str:
        """Generate a sequence of tokens."""
        # Copy the seed tokens.
        tokens = self.seed_tokens[:]
        next_token = None
        while len(tokens) < self.max_tokens and next_token != "$":
            next_token = self.model.generate(random_seed=seed, text_seed=tokens)
            tokens.append(next_token)

        if self.tokenization == "characters":
            tokens = "".join(tokens)
        elif self.tokenization == "words":
            tokens = " ".join(tokens)
        if next_token != "$":
            tokens += " $"

        if not self.quiet:
            logger.info(f"\t{tokens}")

        return tokens

    def generate(self, n: int = None) -> pd.DataFrame:
        if not self.quiet:
            logger.info("Generating haiku...")
        n = n or self.number
        seeds = [self.seed or random.randint(0, 2 ** 32 - 1) for _ in range(n)]
        haiku = [self._generate(s) for s in seeds]

        columns = {
            "model": [self.name] * n,
            "type": [self.type] * n,
            "seed": seeds,
            "haiku": haiku,
        }
        return pd.DataFrame(columns)

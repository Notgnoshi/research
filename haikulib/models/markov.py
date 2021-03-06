import logging
import pathlib
import pickle
import random

import nltk
import nltk.lm
import pandas as pd

from ..data import get_bag_of
from ..nlp import preprocess
from .base import LanguageModel

logger = logging.getLogger(__name__)


class MarkovModel(LanguageModel):
    """A Kneser-Ney smoothed Markov language model."""

    def __init__(self, config: dict, quiet=True):
        super().__init__(config, quiet)
        self.order = config["order"]
        self.tokenization = config["tokenization"]

        self.bag = get_bag_of(kind=self.tokenization, add_tags=True)
        self.vocab = nltk.lm.Vocabulary(self.bag)
        self.model = nltk.lm.models.KneserNeyInterpolated(order=self.order, vocabulary=self.vocab)

    def _serialize(self, directory: pathlib.Path):
        """Save the trained model to disk."""
        filename = directory / (self.name + ".model")
        with open(filename, "wb") as fd:
            pickle.dump(self.model, fd)
        return True

    def _deserialize(self, directory: pathlib.Path):
        """Load the trained model from disk."""
        filename = directory / (self.name + ".model")
        with open(filename, "rb") as fd:
            self.model = pickle.load(fd)
        return True

    def train(self):
        logger.info("Training model...")
        logger.info("tokenizing...")

        corpus = " ".join(self.df["haiku"])
        if self.tokenization == "words":
            tokens = nltk.word_tokenize(corpus)
        elif self.tokenization == "characters":
            tokens = list(corpus)
        ngrams = nltk.everygrams(tokens, max_len=self.order)
        logger.info("fitting...")
        self.model.fit([ngrams], vocabulary_text=self.vocab)
        logger.info("Trained model.")

    def _generate(self, seed) -> str:
        """Generate a sequence of tokens."""
        # Copy the seed tokens.
        if self.tokenization == "words":
            tokens = self.prompt.split()
        elif self.tokenization == "characters":
            tokens = list(self.prompt)

        next_token = None
        while len(tokens) < self.max_tokens and next_token != "$":
            next_token = self.model.generate(random_seed=seed, text_seed=tokens)
            tokens.append(next_token)

        if self.tokenization == "characters":
            tokens = "".join(tokens)
        elif self.tokenization == "words":
            tokens = " ".join(tokens)

        tokens = preprocess(tokens)

        logger.info("Generated: %s", tokens)

        return tokens

    def generate(self, n: int = None) -> pd.DataFrame:
        logger.info("Generating haiku...")
        n = n or self.number
        # The random number generator is seeded already at this point,
        # but we want to avoid generating N of the same haiku.
        # So we use the seeded generator to generate N seeds, and then generate N haiku from those seeds.
        # This way the results are still reproducible.
        seeds = [random.randint(0, 2 ** 32 - 1) for _ in range(n)]
        haiku = [self._generate(s) for s in seeds]

        columns = {
            "model": [self.name] * n,
            "type": [self.type] * n,
            "seed": seeds,
            "prompt": [self.prompt] * n,
            "haiku": haiku,
        }
        return pd.DataFrame(columns)

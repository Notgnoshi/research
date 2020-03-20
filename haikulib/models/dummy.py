import logging
import pathlib
import random

import pandas as pd

from .base import LanguageModel

logger = logging.getLogger(__name__)


class DummyModel(LanguageModel):
    def __init__(self, config: dict, quiet=True):
        super().__init__(config, quiet)
        self.type = "dummy"

    def train(self):
        logger.info("trained.")

    def generate(self, n: int = None) -> pd.DataFrame:
        haiku = ["summer / all these extra prayers / at the dead puppies"]
        model = [self.config["name"]] * len(haiku)
        model_type = [self.type] * len(haiku)

        columns = {
            "model": model,
            "type": model_type,
            "seed": [self.seed or random.randint(0, 2 ** 32 - 1) for _ in range(len(haiku))],
            "prompt": [self.prompt] * len(haiku),
            "haiku": haiku,
        }
        return pd.DataFrame(columns)

    def _serialize(self, directory: pathlib.Path):
        filename = directory / (self.name + "model")
        filename.touch()
        return filename.exists()

    def _deserialize(self, directory: pathlib.Path):
        filename = directory / (self.name + "model")
        return filename.exists()

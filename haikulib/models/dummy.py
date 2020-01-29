import pathlib

import pandas as pd

from .base import LanguageModel


class DummyModel(LanguageModel):
    def __init__(self, config: dict, quiet=True):
        super().__init__(config, quiet)
        self.type = "dummy"

    def train(self):
        print("trained.")

    def generate(self, n: int = None) -> pd.DataFrame:
        haiku = ["summer / all these extra prayers / at the dead puppies"]
        model = [self.config["name"]] * len(haiku)
        model_type = [self.type] * len(haiku)

        columns = {
            "model": model,
            "type": model_type,
            "seed": [self.seed or random.randint(0, 2 ** 32 - 1) for _ in range(len(haiku))],
            "haiku": haiku,
        }
        return pd.DataFrame(columns)

    def _serialize(self, filename: pathlib.Path):
        filename.touch()
        return filename.exists()

    def _deserialize(self, filename: pathlib.Path):
        return filename.exists()

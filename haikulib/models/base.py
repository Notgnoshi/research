import abc
import logging
import pathlib
import pprint
from typing import Union

import commentjson
import pandas as pd

from ..data import get_df

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    pass


class LanguageModel(abc.ABC):
    """An abstract base language model."""

    def __init__(self, config: dict, quiet=True):
        self.quiet = quiet

        try:
            self._validate(config)
        except ConfigValidationError:
            raise ValueError("Provided model configuration is invalid.")

        self.config = config
        self.name = config["name"]
        self.prompt = config["prompt"]
        self.seed = config["seed"]
        self.tags = config["tags"]
        self.type = config["type"]

        self.df = get_df()

        # The number of haiku to generate.
        self.number = config["number"]
        # The max number of tokens to generate.
        self.max_tokens = config["max_tokens"]

    @abc.abstractmethod
    def train(self):
        """Train the language model using the configured parameters."""

    @abc.abstractmethod
    def generate(self, n: int = None) -> pd.DataFrame:
        """Generate text using the configured parameters.

        The generated DataFrame should have columns "model", "type", "seed", and "haiku",
        in that order. The "model" column is the model name. "type" is the type of model,
        "seed" is the random seed used, and "haiku" is the generated haiku.
        """

    @abc.abstractmethod
    def _serialize(self, directory: pathlib.Path):
        """Model-specific serialization to disk."""

    @abc.abstractmethod
    def _deserialize(self, directory: pathlib.Path):
        """Model-specific deserialization from disk."""

    def serialize(self, directory: Union[pathlib.Path, str] = None):
        """Save a trained language model to a file."""
        directory = directory or self.config["output_directory"]
        if not self.quiet:
            logger.info("Saving model to %s...", directory)
        success = self._serialize(pathlib.Path(directory))
        if not success and not self.quiet:
            logger.info("Failed to save model.")
        elif not self.quiet:
            logger.info("Saved model.")

    def save(self, df: pd.DataFrame, filename: Union[pathlib.Path, str] = None):
        """Save the generated content to a file."""
        filename = filename or self.config["generated_path"]

        if not self.quiet:
            logger.info("Saving generated text to %s...", filename)
            logger.info(df)

        # TODO: Validate the DataFrame before appending to an existing CSV file.
        # TODO: It might be necessary to read in the file into a DataFrame before appending.

        with open(filename, "a") as f:
            # Only add the header if the file is empty
            # Don't use an index column, so that we can append at will to the csv.
            # Requires index_col=False passed to pd.read_csv.
            df.to_csv(f, mode="a", header=(f.tell() == 0), index=False)

    def deserialize(self, directory: Union[pathlib.Path, str] = None):
        """Load a trained language model from a file."""
        directory = directory or self.config["output_directory"]
        if not self.quiet:
            logger.info("Loading model from %s...", directory)
        success = self._deserialize(pathlib.Path(directory))
        if not success and not self.quiet:
            logger.info("Failed to load model.")
        elif not self.quiet:
            logger.info("Loaded model.")

    @staticmethod
    def read_config(path: Union[pathlib.Path, str]):
        """Read a JSON-with-comments configuration file from disk."""
        path = pathlib.Path(path)
        if not path.exists():
            raise ConfigValidationError(f"Model configuration '{path}' not found.")
        with open(path, "r") as file:
            config = commentjson.load(file)

        path = path.resolve()

        if "output_directory" not in config:
            config["output_directory"] = path.parent
        else:
            config["output_directory"] = pathlib.Path(config["output_directory"]).resolve()

        if "name" not in config:
            config["name"] = path.stem

        if "generated_path" not in config or config["generated_path"] is None:
            config["generated_path"] = config["output_directory"] / (config["name"] + "csv")
            config["generated_path"] = config["generated_path"].resolve()
        else:
            config["generated_path"] = pathlib.Path(config["generated_path"]).resolve()

        return config

    @staticmethod
    def _validate(config: dict):
        """Validate the given language model parameters.

        Throws an exception if the configuration parameters aren't valid.

        TODO: Override in the base classes.

        :param config: The model parameters
        """
        logger.info(pprint.pformat(config))

        if config["type"] == "markov":
            if "tokenization" not in config:
                raise ConfigValidationError("Markov models must specify tokenization")
            if config["tokenization"] not in ("words", "characters"):
                raise ConfigValidationError("Tokenization must be one of 'words' or 'characters'")
        if config["type"] == "transformer":
            if "model" not in config:
                raise ConfigValidationError("Transformer models must specify the specific model.")
            if config["model"] not in ("gpt", "gpt2", "bert", "distilbert", "camembert", "roberta"):
                raise ConfigValidationError("Unknown transformer model.")

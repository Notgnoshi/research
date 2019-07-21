#!/usr/bin/env python3
import argparse
import pathlib
import sys


REPO_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR: pathlib.Path = REPO_DIR / "data"
TGRNN_DIR: pathlib.Path = REPO_DIR / "depends" / "textgenrnn"
EXPERIMENT_DIR: pathlib.Path = DATA_DIR / "experiments" / "generation"

# Use own, modified version of textgenrnn.
# sys.path.insert(0, str(TGRNN_DIR))
sys.path.append(str(TGRNN_DIR))
from textgenrnn import textgenrnn

# TODO: BFS search **kwargs and come up with all of the tweakable parameters.
DEFAULT_MODEL_CONFIG = {
    "rnn_size": 128,
    "rnn_layers": 4,
    "rnn_bidirectional": True,
    "max_length": 40,
    "max_words": 10000,
    "dim_embeddings": 100,
    "word_level": False,
}

DEFAULT_TRAIN_CONFIG = {
    "line_delimited": False,
    "num_epochs": 10,
    "gen_epochs": 2,
    "batch_size": 1024,
    "train_size": 0.8,
    "dropout": 0.0,
    "max_gen_length": 200,
    "validation": False,
    "is_csv": False,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Use Max Woolf's textgenrnn to generate haikus."""
    )

    parser.add_argument("--name", default="textgenrnn-initial", help="Textgenrnn model name.")

    model = parser.add_argument_group()
    model.add_argument("--rnn-size", type=int, default=128)
    model.add_argument("--rnn-layers", type=int, default=4)
    model.add_argument("--rnn-bidirectional", action="store_true", default=False)
    model.add_argument(
        "--max-length", type=int, default=40, help="Maximum number of tokens per seed"
    )
    model.add_argument("--dim-embeddings", type=int, default=100)
    model.add_argument(
        "--word-level", action="store_true", default=False, help="Use word level tokenization"
    )

    train = parser.add_argument_group()
    train.add_argument("--num-epochs", type=int, default=10)
    train.add_argument("--gen-epochs", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument(
        "--train-size", type=float, default=0.8, help="The training percentage of the dataset."
    )
    train.add_argument("--dropout", type=float, default=0.0)
    train.add_argument("--max-gen-length", type=int, default=200)
    train.add_argument("--validation", action="store_true", default=False)

    return parser.parse_args()


def main(args):
    # TODO: Load model from file if the right commandline options have been given.
    model = textgenrnn(weights_path=None, vocab_path=None, config_path=None, name=args.name)

    model.train_from_file(
        DATA_DIR / "cleaned.txt",
        header=False,
        new_model=True,
        context=None,
        is_csv=False,
        num_epochs=args.num_epochs,
        gen_epochs=args.gen_epochs,
        batch_size=args.batch_size,
        train_size=args.train_size,
        dropout=args.dropout,
        max_gen_length=args.max_gen_length,
        validation=args.validation,
        save_epochs=0,
        multi_gpu=False,
        verbose=1,
    )

    # TODO: Save model weights, vocab, and config.

    # TODO: model parameters?
    model.generate(15)


if __name__ == "__main__":
    if not EXPERIMENT_DIR.exists():
        EXPERIMENT_DIR.mkdir(exist_ok=True, parents=True)

    main(parse_args())

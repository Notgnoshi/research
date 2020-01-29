#!/usr/bin/env python3
"""Train Markov models to generate haiku."""
import argparse
import pathlib

import haikulib.data
from haikulib.models import MarkovModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--quiet", "-q", action="store_true", default=False, help="Eliminate console output",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=pathlib.Path,
        default=haikulib.data.get_data_dir() / "models" / "default-markov.jsonc",
        help="Configuration file to use.",
    )

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--train",
        "-t",
        action="store_true",
        default=False,
        help="Whether to train a langage model.",
    )
    action.add_argument(
        "--generate",
        "-g",
        action="store_true",
        default=False,
        help="Whether to use a trained language model to generate text.",
    )

    return parser.parse_args()


def main(args):
    config = MarkovModel.read_config(args.config)
    model = MarkovModel(config, args.quiet)

    if args.train:
        model.train()
        model.serialize()
    elif args.generate:
        model.deserialize()
        df = model.generate()
        model.save(df)


if __name__ == "__main__":
    main(parse_args())

#!/usr/bin/env python3
import argparse
import sys

from datasets import DATASETS


def validate(datasets):
    """Validate the given datasets against the known datasets.

    :param datasets: The datasets to validate.
    :type datasets: A list of str dataset IDs.
    :returns: A list of validated dataset IDs.
    """
    ids = []
    for key in datasets:
        key = key.lower()
        if key not in DATASETS:
            print("dataset '{}' unknown. Skipping.".format(key), file=sys.stderr)
        else:
            ids.append(key)
    return ids


def parse_args():
    """Parse and return commandline arguments."""
    parser = argparse.ArgumentParser(description="Download datasets for this research project.")
    parser.add_argument(
        "datasets",
        type=str,
        nargs="+",
        help="A space-separated list of datasets to download. If 'all' is given, all available "
        "datasets will be downloaded. One of: {}".format(list(DATASETS.keys())),
    )

    return parser.parse_args()


def main(args):
    """Download datasets for this research project."""
    if "all" in args.datasets:
        args.datasets = list(DATASETS.keys())

    args.datasets = validate(args.datasets)

    for dataset in args.datasets:
        dataset = DATASETS[dataset]
        if not dataset.exists():
            dataset.download()

if __name__ == "__main__":
    main(parse_args())

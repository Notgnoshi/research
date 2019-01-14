#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import requests

DATASETS = {"nietzsche": ("https://s3.amazonaws.com/text-datasets/nietzsche.txt", "nietzsche.txt")}


def curl(url, filename):
    """Download a file from the internet.

    Overwrites the file if it exists.

    :param url: The URL to Download
    :type url: str
    :param filename: The filename to save the file as
    :type filename: str
    """
    r = requests.get(url, stream=True)
    with open(filename, "wb") as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)


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


def download(key):
    """Download the dataset corresponding to the given key.

    :param key: The dataset id
    :type key: str
    """
    url, filename = DATASETS[key]
    # Download the files in the data/ directory
    filename = Path(__file__).parent.joinpath(Path(filename))
    print("Downloading {} from {}...".format(filename, url))
    curl(url, filename)


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
        download(dataset)


if __name__ == "__main__":
    main(parse_args())

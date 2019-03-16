"""Helper functions for loading data into multiple different representations."""
from ast import literal_eval
from collections import Counter
from pathlib import Path

import pandas as pd


def get_df():
    """Get the dataset unmodified in a pandas.DataFrame."""
    return pd.read_csv(
        Path(__file__).parent.parent.parent.joinpath("data/haikus.csv"),
        index_col=0,
        # Ensure that the list of lines is interpreted as a list, not a string...
        converters={"haiku": literal_eval},
    )


def get_bag_of_words():
    """Get the dataset in a bag of words representation."""
    df = get_df()
    bag = Counter()
    for haiku in df["haiku"]:
        for line in haiku:
            bag.update(line.split())

    return bag


def get_bag_of_lines():
    """Get the dataset in a bag of lines representation."""
    df = get_df()
    lines = []
    for haiku in df["haiku"]:
        for line in haiku:
            lines.append(line)
    return Counter(lines)

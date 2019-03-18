"""Helper functions for loading data into multiple different representations."""
import re
import string
from ast import literal_eval
from collections import Counter
from pathlib import Path

import pandas as pd

# Preserve alphabetic, numeric, spaces, and single quotes.
ALPHABET = frozenset(string.ascii_lowercase + " " + "'" + "/" + string.digits)


def preprocess(text):
    """Preprocess the text of a haiku.

    Remove all punctuation, except for apostophes.
    Ensure ascii character set.
    """
    # Replace 0xa0 (unicode nbsp) with an ascii space
    text = text.replace("\xa0", " ")
    # Replace hyphen and en dash with space
    text = text.replace("-", " ")
    text = text.replace("–", " ")
    # Remove all other non-ascii characters
    text = text.encode("ascii", "ignore").decode("utf-8")
    # Remove redundant whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return "".join(filter(ALPHABET.__contains__, text.lower()))


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


def read_from_file():
    """Get a list of unclean haikus from the text file."""
    haikus = []
    with open(Path(__file__).parent.parent.parent.joinpath("data/haikus.txt"), "r") as datafile:
        haiku = ""
        for line in datafile:
            line = line.strip()
            if line:
                if haiku:
                    haiku += " / "
                haiku += line
            elif not line and haiku:
                haikus.append(haiku)
                haiku = ""
    return haikus

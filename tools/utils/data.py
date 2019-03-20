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
        # Ensure that the lists of lines are interpreted as a list, not a string...
        converters={"haiku": literal_eval, "nostopwords": literal_eval, "lemmas": literal_eval},
    )


def get_bag_of(kind):
    """Get a bag of ... representation of the dataset.

    Kind may be one of 'words', 'lines', 'nostopwords', or 'lemmas'.
    """
    df = get_df()
    bag = Counter()

    if kind == "lines":
        lines = []
        for haiku in df["haiku"]:
            for line in haiku:
                lines.append(line)
        bag = Counter(lines)

    columns = {
        # token kind: column name
        "words": "haiku",
        "lemmas": "lemmas",
        "nostopwords": "nostopwords",
    }

    if kind in ("words", "lemmas", "nostopwords"):
        column = columns[kind]
        for haiku in df[column]:
            for line in haiku:
                bag.update(line.split())
    else:
        raise ValueError(f"bag of '{kind}' is unsupported.")

    return bag


def get_bag_of_words():
    """Get the dataset in a bag of words representation."""
    return get_bag_of("words")


def get_bag_of_lines():
    """Get the dataset in a bag of lines representation."""
    return get_bag_of("lines")


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

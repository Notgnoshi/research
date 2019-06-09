"""Helper functions for loading data into multiple different representations."""
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd

# Preserve alphabetic, numeric, spaces, and single quotes.
ALPHABET = frozenset(string.ascii_lowercase + " " + "'" + "/" + "#" + string.digits)


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
        Path(__file__).parent.parent.parent.parent.joinpath("data/haikus.csv"), index_col=0
    )


def __get_bag_of_words(df, column):
    """Get a bag of words representation of given column from the dataframe."""
    bag = Counter()
    for haiku in df[column]:
        # The /'s are separated by space on each side, so they get tokenized as their own symbol.
        bag.update(haiku.split())

    # Do not count the line separator as a word.
    if "/" in bag:
        del bag["/"]

    return bag


def __get_bag_of_lines(df, column):
    all_lines = []
    for haiku in df[column]:
        lines = haiku.split("/")
        lines = [l.strip() for l in lines]
        all_lines += lines

    return Counter(lines)


def get_bag_of(column, kind):
    """Get a bag of 'kind' representation of the dataset.

    :param column: The dataset column to use.
    :param kind: The kind of bag. One of 'words' or 'lines'.
    :rtype: collections.Counter
    """
    if kind not in ("words", "lines"):
        raise ValueError('kind "{}" is unsupported.'.format(kind))
    if column not in ("haiku", "lemmas", "nostopwords"):
        raise ValueError('column "{}" is unsupported'.format(column))

    df = get_df()

    if kind == "words":
        return __get_bag_of_words(df, column)

    return __get_bag_of_lines(df, column)


def read_from_file():
    """Get a list of unclean haiku from the text file.

    Each haiku is a single string, with lines separated by `/`.
    """
    haikus = []
    with open(
        Path(__file__).parent.parent.parent.parent.joinpath("data/haikus.txt"), "r"
    ) as datafile:
        haiku = ""
        for line in datafile:
            line = line.strip()
            if line:
                if haiku:
                    # Separate the /'s by spaces so that str.split() works more intuitively.
                    haiku += " / "
                haiku += line
            elif not line and haiku:
                # Add an end-of-haiku symbol.
                haiku += " #"
                haikus.append(haiku)
                haiku = ""
    return haikus

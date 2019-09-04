"""Helper functions for loading data into multiple different representations."""
import collections
import pathlib
import re
import string

import pandas as pd

# Preserve alphabetic, numeric, spaces, and single quotes.
ALPHABET = frozenset(string.ascii_lowercase + " " + "'" + "/" + "#" + string.digits)


def get_data_dir() -> pathlib.Path:
    """Get the path to the data directory in this repository."""
    return pathlib.Path(__file__).parent.parent.parent.parent.joinpath("data").resolve()


def read_from_file() -> list:
    """Get a list of unclean haiku from the text file.

    Each haiku is a single string, with lines separated by `/`, and an end-of-haiku symbol `#`.
    """
    haikus = []
    with open(get_data_dir() / "haiku.txt", "r") as datafile:
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


def preprocess(text: str) -> str:
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


def init_data_dir():
    """Initialize the data directory so that the right things exist.

    Preprocesses the haiku, and writes them to a CSV file.
    """
    haiku = read_from_file()
    haiku = (preprocess(h) for h in haiku)
    # Removes duplicates in a manner that preserves order. Requires Python 3.6+
    haiku = list(dict.fromkeys(haiku))
    lines = [h.count("/") + 1 for h in haiku]

    # TODO: Add EDA classifiers as they are written, such as occurrences of colors in each haiku.
    rows = {"haiku": haiku, "lines": lines}

    df = pd.DataFrame(rows)
    df.to_csv(get_data_dir() / "haiku.csv")


def get_df(init=False) -> pd.DataFrame:
    """Get the dataset unmodified in a pandas.DataFrame."""
    if init:
        init_data_dir()

    return pd.read_csv(get_data_dir() / "haiku.csv", index_col=0)


def __get_bag_of_words(df, column):
    """Get a bag of words representation of given column from the dataframe."""
    bag = collections.Counter()
    for haiku in df[column]:
        # The /'s are separated by space on each side, so they get tokenized as their own symbol.
        bag.update(haiku.split())

    # Do not count the line or haiku separators as a words.
    if "/" in bag:
        del bag["/"]
    if "#" in bag:
        del bag["#"]

    return bag


def __get_bag_of_lines(df, column):
    all_lines = []
    for haiku in df[column]:
        lines = haiku.split("/")
        lines = [l.strip(" \t\n#") for l in lines]
        all_lines += lines

    return collections.Counter(lines)


def get_bag_of(column, kind) -> collections.Counter:
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


def tokenize(haiku, method="words") -> list:
    """Tokenize the given haiku using the specified method.

    :param haiku: The (single) haiku to tokenize.
    :type haiku: str, of the form 'line / line #'
    :param method: The tokenization method. One of 'words' or 'characters', defaults to 'words'
    """
    if method == "words":
        return haiku.split()
    elif method == "characters":
        # Strings are already tokenized into characters. Duh...
        return list(haiku)

    raise ValueError(f"Unrecognized tokenization method '{method}'")


def get_flowers() -> set:
    """Get a set of flower names."""
    path = get_data_dir() / "flora.txt"
    with path.open("r") as f:
        flowers = set(preprocess(line) for line in f)

    return flowers


def get_colors() -> pd.DataFrame:
    """Get a dataframe of color -> HTML colors.

    Note that this CSV file uses hex RGB color codes for many of the colors, but falls back to using
    HTML named colors for colors without an RGB value.

    The colors with RGB values came from https://xkcd.com/color/rgb/ while the colors with the named
    values came from
    https://medium.com/@eleanorstrib/python-nltk-and-the-digital-humanities-finding-patterns-in-gothic-literature-aca84639ceeb
    """
    return pd.read_csv(get_data_dir() / "colors.csv", index_col=0)


def get_colors_dict() -> dict:
    """Get a dictionary of color -> HTML color mappings."""
    df = get_colors()
    return {row["color"]: row["rgb"] for row in df}


def get_animals() -> set:
    """Get a set of animal names.

    TODO: The fauna dataset has many flora species listed. Remove flora.
    """
    path = get_data_dir() / "fauna.txt"
    with path.open("r") as f:
        animals = set(preprocess(line) for line in f)

    return animals

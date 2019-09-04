"""Helpers for accessing data from the prepared datasets."""
import ast
import collections
import pathlib

import pandas as pd

from haikulib.nlp import preprocess


def get_data_dir() -> pathlib.Path:
    """Get the path to the data directory in this repository."""
    return pathlib.Path(__file__).parent.parent.parent.joinpath("data").resolve()


def get_df() -> pd.DataFrame:
    """Get the dataset in a nicely structured DataFrame."""
    return pd.read_csv(
        get_data_dir() / "haiku.csv", index_col=0, converters={"colors": ast.literal_eval}
    )


def __get_bag_of_words(df: pd.DataFrame) -> collections.Counter:
    bag = collections.Counter()
    for haiku in df["haiku"]:
        # The /'s are separated by space on each side, so they get tokenized as their own symbol.
        bag.update(haiku.split())

    # Do not count the line or haiku separators as a words.
    if "/" in bag:
        del bag["/"]
    if "#" in bag:
        del bag["#"]

    return bag


def __get_bag_of_lines(df: pd.DataFrame) -> collections.Counter:
    all_lines = []
    for haiku in df["haiku"]:
        lines = haiku.split("/")
        lines = [l.strip(" \t\n#") for l in lines]
        all_lines += lines

    return collections.Counter(lines)


def get_bag_of(kind: str) -> collections.Counter:
    """Get a bag of 'kind' representation of the dataset.

    :param kind: The kind of bag. One of 'words' or 'lines'.
    """
    df = get_df()

    if kind == "words":
        return __get_bag_of_words(df)
    if kind == "lines":
        return __get_bag_of_lines(df)

    raise ValueError('kind "{}" is unsupported.'.format(kind))


def get_flowers() -> set:
    """Get a set of flower names."""
    path = get_data_dir() / "flora.txt"
    with path.open("r") as f:
        flowers = set(preprocess(line) for line in f)

    return flowers


def get_colors() -> pd.DataFrame:
    """Get a DataFrame of color -> HTML colors.

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
    return {row["color"]: row["rgb"] for index, row in df.iterrows()}


def get_animals() -> set:
    """Get a set of animal names.

    TODO: The fauna dataset has many flora species listed. Remove flora.
    """
    path = get_data_dir() / "fauna.txt"
    with path.open("r") as f:
        animals = set(preprocess(line) for line in f)

    return animals

"""Helpers for accessing data from the prepared datasets."""
import ast
import collections
import pathlib
import random

import pandas as pd

from haikulib.nlp import preprocess


def get_data_dir() -> pathlib.Path:
    """Get the path to the data directory in this repository."""
    return pathlib.Path(__file__).parent.parent.parent.joinpath("data").resolve()


def get_df() -> pd.DataFrame:
    """Get the dataset in a nicely structured DataFrame."""
    return pd.read_csv(
        get_data_dir() / "haiku.csv",
        index_col=0,
        converters={"colors": ast.literal_eval, "syllables": ast.literal_eval},
    )


def get_generated_df(path: pathlib.Path = None) -> pd.DataFrame:
    """Get the generated haiku."""
    if path is None:
        path = get_data_dir() / "generated.csv"

    if not path.exists():
        return None

    return pd.read_csv(path, index_col=0)


def get_random_prompt():
    """Pick a random prompt from the haiku dataset."""
    df = get_df()
    prompts = []
    for haiku in df["haiku"]:
        prompts.append(haiku.split()[0])
    return random.sample(prompts, 1)[0]


def __get_bag_of_characters(df: pd.DataFrame) -> collections.Counter:
    bag = collections.Counter()
    for haiku in df["haiku"]:
        bag.update(haiku)
    return bag


def __get_bag_of_words(df: pd.DataFrame) -> collections.Counter:
    bag = collections.Counter()
    for haiku in df["haiku"]:
        # The /'s are separated by space on each side, so they get tokenized as their own symbol.
        bag.update(haiku.split())
    return bag


def __get_bag_of_lines(df: pd.DataFrame) -> collections.Counter:
    all_lines = []
    for haiku in df["haiku"]:
        lines = haiku.split("/")
        lines = [l.strip(" \t\n") for l in lines]
        all_lines += lines

    return collections.Counter(lines)


def get_bag_of(kind: str, add_tags: bool) -> collections.Counter:
    """Get a bag of 'kind' representation of the dataset.

    :param kind: The kind of bag. One of 'words' or 'lines'.
    """
    df = get_df()

    if add_tags:
        for idx, row in df.iterrows():
            haiku = "^ " + row["haiku"] + " $"
            df.at[idx, "haiku"] = haiku

    if kind == "characters":
        return __get_bag_of_characters(df)
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


def get_animals() -> set:
    """Get a set of animal names.

    TODO: The fauna dataset has many flora species listed. Remove flora.
    """
    path = get_data_dir() / "fauna.txt"
    with path.open("r") as f:
        animals = set(preprocess(line) for line in f)

    return animals

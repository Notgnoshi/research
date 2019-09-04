"""Helpers for preparing the haiku dataset.

The purpose of these functions is to avoid saving the haiku in a very large CSV file in version
control. The CSV file is prone to change often, both in size and structure, due to different
metadata fields being added as exploratory data analysis is completed.
"""
import pandas as pd

from haikulib.data import get_data_dir
from haikulib.eda.colors import find_colors
from haikulib.nlp import pos_tag, preprocess


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


def init_csv():
    """Initialize the data directory so that the right things exist.

    Preprocesses the haiku, and writes them to a CSV file.
    """
    haiku = read_from_file()
    haiku = (preprocess(h) for h in haiku)
    # Removes duplicates in a manner that preserves order. Requires Python 3.6+
    haiku = list(dict.fromkeys(haiku))
    lines = [h.count("/") + 1 for h in haiku]

    rows = {"haiku": haiku, "colors": [find_colors(pos_tag(h)) for h in haiku], "lines": lines}

    df = pd.DataFrame(rows)
    df.to_csv(get_data_dir() / "haiku.csv")

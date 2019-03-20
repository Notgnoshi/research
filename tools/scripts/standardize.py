#!/usr/bin/env python3
import sys
import pathlib
import pandas as pd

root = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(root)
from tools.utils import lemmatize, preprocess, read_from_file, remove_stopwords


def process_haiku(haiku):
    haiku = preprocess(haiku)
    lemmatized = lemmatize(haiku)
    lemmatized = lemmatized.split("/")
    lemmatized = [l.strip() for l in lemmatized]
    nostops = [l.strip() for l in remove_stopwords(haiku).split("/")]
    haiku = [l.strip() for l in haiku.split("/")]

    return {"haiku": haiku, "nostops": nostops, "lemmas": lemmatized, "lines": len(haiku)}


def main():
    haikus = read_from_file()

    # Remove duplicates in a manner that preserves order.
    # Requires Python 3.6+
    haikus = list(dict.fromkeys(haikus))

    rows = list(map(process_haiku, haikus))

    haikus = pd.DataFrame(rows)
    haikus.tail()

    haikus.to_csv("../haikus.csv")


if __name__ == "__main__":
    main()

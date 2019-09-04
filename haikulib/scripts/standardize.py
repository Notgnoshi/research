#!/usr/bin/env python3
import csv
import pathlib
import sys
import warnings

warnings.warn(
    "This script is deprecated in favor of running haikulib.data.init_data_dir()",
    DeprecationWarning,
)

import pandas as pd

REPO_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = REPO_DIR / "data"

# Add repository root directory to path so that haikulib.utils.data is importable.
sys.path.append(str(REPO_DIR))
from haikulib.utils.data import preprocess, read_from_file
from haikulib.utils.nlp import lemmatize, remove_stopwords


def process_haikus(haikus):
    # Produce an iterator of lemmatized haikus, processed in parallel.
    # The rest is fast, so no need to parallelize
    lemmas = lemmatize(haikus)

    # Before splitting each haiku into a list of lines, remove stopwords from each haiku.
    nostops = (remove_stopwords(haiku) for haiku in haikus)
    # Trim whitespace from each haiku.
    nostops = (haiku.strip() for haiku in nostops)

    # Trim whitespace from each haiku.
    haikus = (haiku.strip() for haiku in haikus)

    # Trim whitespace from each haiku.
    lemmas = (lemma.strip() for lemma in lemmas)

    haikus = list(haikus)
    # Count the number of lines for each haiku. Perform after the list conversion
    # to avoid depleting the haikus generator.
    lines = [haiku.count("/") + 1 for haiku in haikus]
    nostops = list(nostops)
    # Since lemmas is an iterator, this is where the actual parallel computation will be performed.
    lemmas = list(lemmas)

    rows = {"haiku": haikus, "lines": lines, "nostopwords": nostops, "lemmatized": lemmas}
    return rows


def main():
    haikus = read_from_file()

    # Preprocess each haiku before doing anything. Doing so before the duplicate removal
    # will result in additional duplicates being removed.
    haikus = [preprocess(haiku) for haiku in haikus]

    # Remove duplicates in a manner that preserves order.
    # Requires Python 3.6+
    haikus = list(dict.fromkeys(haikus))

    rows = process_haikus(haikus)

    haikus = pd.DataFrame(rows)
    print(haikus.tail())

    haikus.to_csv(DATA_DIR / "haiku.csv")
    haikus.to_csv(
        DATA_DIR / "cleaned.txt",
        columns=["haiku"],
        header=False,
        index=False,
        sep="|",
        quoting=csv.QUOTE_NONE,
    )


if __name__ == "__main__":
    main()

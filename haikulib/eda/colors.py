"""Perform exploratory data analysis to parse colors from the haiku dataset.

find_colors() is used to initialize the haiku dataset, while get_color_counts is used to process
the prepared dataset after it's been initialized.
"""
import itertools
from collections import Counter
from typing import Iterable, List, Tuple

import nltk
import pandas as pd

from haikulib.data import get_colors_dict, get_df

COLORS = get_colors_dict()
COLOR_POS_TAGS = frozenset({"JJ", "NN"})


def is_color(tagged_word: Tuple[str, str]) -> bool:
    """Determine if the given word is a color based on its part-of-speech.

    :param tagged_word: A word that's been tagged with nltk.pos_tag()
    """
    word, pos = tagged_word
    return pos in COLOR_POS_TAGS and word in COLORS


def find_colors(text: Iterable[Tuple[str, str]]) -> List[str]:
    """Return an unordered list of colors from the given POS-tagged text.

    Check for 1, 2, and 3-gram colors like "dark blue".

    Attempt to make the 1, 2, 3-grams exclusive so that a text containing "light olive green"
    (#a4be5c) will return just
        ["light olive green"]
    instead of
        ["light", "olive", "green", "olive green", "light olive green"]

    :param text: The POS-tagged text to search for colors.
    :return: A list of colors appearing in the provided text.
    """
    colors = []

    # Pad the right of any text that is too short to prevent a nasty crash.
    ngrams = nltk.ngrams(text, n=3, pad_right=True, right_pad_symbol=("?", "??"))
    for ngram in ngrams:
        word = " ".join(w[0] for w in ngram)
        # Check the 3-gram
        if word in COLORS:
            colors.append(word)
            # Skip over the rest of this ngram.
            next(ngrams)
            next(ngrams)
        # If the 3-gram wasn't a color, check the 2-gram.
        else:
            word = " ".join(w[0] for w in ngram[:2])
            if word in COLORS:
                colors.append(word)
                # Skip over the rest of this ngram.
                next(ngrams)
            # If the 2-gram wasn't a color, check the 1-gram, using the tagged part-of-speech.
            elif is_color(ngram[0]):
                colors.append(ngram[0][0])

    try:
        # Check the last 2-gram and the last two 1-grams by hand (skipped by loop)
        if ngram[1:] in COLORS:
            word = " ".join(w[0] for w in ngram[1:])
            colors.append(word)
        else:
            if is_color(ngram[-2]):
                colors.append(ngram[-2][0])
            if is_color(ngram[-1]):
                colors.append(ngram[-1][0])
    except UnboundLocalError:
        # As with life, problems are best left ignored.
        pass

    return colors


def get_color_counts() -> pd.DataFrame:
    """Get a DataFrame of the color counts and HTML color codes."""
    df = get_df()
    colors = Counter(itertools.chain.from_iterable(df["colors"].tolist()))

    # Relies on dicts being sorted, added in Python 3.6, guaranteed by Python 3.7
    return pd.DataFrame(
        {
            "color": list(colors.keys()),
            "count": list(colors.values()),
            "html_color": [COLORS[c] for c in colors],
        }
    )

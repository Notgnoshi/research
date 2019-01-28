#!/usr/bin/env python3
import pickle
import string
from pathlib import Path

ALPHABET = frozenset(string.ascii_lowercase + " ")


def preprocess(text):
    """Clean the given text.

    Leaves only lowercase ascii letters and spaces.

    :param text: The text to clean.
    :type text: str
    :return: The cleaned text.
    :rtype: str
    """
    return "".join(filter(ALPHABET.__contains__, text.strip().lower()))


def load(filename):
    """Load a pickled object from the given filename.

    :param filename: The pickled object filename.
    :return: whatever was pickled
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def long_enough(haiku):
    """Determine if the given haiku is long enough to be a haiku."""
    # There should be 4 or more words in the haiku
    if haiku.count(" ") < 3:
        return False

    return True


if __name__ == "__main__":
    haikus = Path(__file__).parent.parent.joinpath("haikus.pkl")
    haikus = load(haikus)

    cleaned = []

    for source in haikus:
        cleaned += [preprocess(h) for h in haikus[source]]

    # Remove some of the false positives.
    cleaned[:] = [x for x in cleaned if long_enough(x)]

    # Remove duplicates.
    cleaned = set(cleaned)
    cleaned = list(cleaned)

    print("Saving", len(cleaned), "cleaned haikus in cleaned.pkl")
    with open(Path(__file__).parent.parent.joinpath("cleaned.pkl"), "wb") as f:
        pickle.dump(cleaned, f)

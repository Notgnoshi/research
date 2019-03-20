#!/usr/bin/env python3
import pickle
import re
import string
from pathlib import Path

ALPHABET = frozenset(string.ascii_lowercase + " ")
REGEX = re.compile(r"[^a-zA-Z]", flags=re.IGNORECASE)


def preprocess(text):
    """Clean the given text.

    Leaves only lowercase ascii letters and spaces.

    :param text: The text to clean.
    :type text: str
    :return: The cleaned text.
    :rtype: str
    """
    # Replace non alphabetic things with spaces.
    text, _ = re.subn(REGEX, " ", text)
    # Replace multiple spaces with a single space.
    text = " ".join(text.split())
    # Re-add apostrophes.
    text = text.replace(" s ", "'s ")
    text = text.replace(" t ", "'t ")
    # Lowercase the text.
    return text.lower()


def load(filename):
    """Load a pickled object from the given filename.

    :param filename: The pickled object filename.
    :return: whatever was pickled
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def is_haiku(haiku):
    """Determine if the given haiku might actually be a haiku."""
    # There should be 6 or more words in the haiku
    if haiku.count(" ") < 5:
        return False

    # Avoid catching the list of volumes.
    if "volume" in haiku:
        return False

    if "nbsp" in haiku:
        return False

    # Determined by hand.
    if len(haiku) > 99 or len(haiku) < 26:
        return False

    return True


if __name__ == "__main__":
    haikus = Path(__file__).parent.parent.parent.joinpath("data/haikus.pkl")
    haikus = load(haikus)

    cleaned = []

    for source in haikus:
        cleaned += [preprocess(h) for h in haikus[source]]

    # Remove some of the false positives.
    cleaned[:] = [x for x in cleaned if is_haiku(x)]

    # Remove duplicates.
    cleaned = set(cleaned)
    cleaned = list(cleaned)

    print("Saving", len(cleaned), "cleaned haikus in cleaned.pkl")
    with open(Path(__file__).parent.parent.parent.joinpath("data/cleaned.pkl"), "wb") as f:
        pickle.dump(cleaned, f)

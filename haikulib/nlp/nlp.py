"""Helpers for common NLP tasks."""
import re
import string
from collections import Counter
from typing import Dict, Generator, Iterable, Sequence, Set, Tuple, Union

import nltk
import spacy
from nltk.corpus import stopwords as nltk_stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
# Preserve alphabetic, numeric, spaces, and single quotes.
ALPHABET = frozenset(string.ascii_lowercase + " " + "'" + "/" + string.digits)


def __stop_words():
    """Combine the list of stop words from Spacy and NLTK."""
    return frozenset().union(nltk_stop_words.words("english"), spacy_stop_words)


STOPWORDS = __stop_words()


def remove_stopwords(text: str) -> str:
    """Remove stopwords from the given text."""
    return " ".join(word for word in text.split() if word not in STOPWORDS)


def lemmatize(texts: Iterable[str]) -> Generator[str, None, None]:
    """Return the lemmatized version of the given text."""
    for doc in nlp.pipe(texts, batch_size=512, n_threads=-1):
        yield " ".join(token.lemma_ for token in doc)


def pos_tag(text: str) -> Iterable[Tuple[str, str]]:
    """Part-of-speech tag the given text.

    :param text: The text to tag
    """
    return nltk.pos_tag([w for w in text.split() if w not in {"/"}])


def count_tokens_from(text: str, sentinels: Union[Set, Dict], ngrams: Sequence = None):
    """Count tokens in `text` that are in the set of sentinels.

    If ngrams is given, consider ngrams as well.

    :param text: The text to search for sentinels.
    :param sentinels: The sentinels to count occurrences of.
    :param ngrams: A sequence of ngram sizes, defaults to None
    """
    ngrams = range(1) if ngrams is None else ngrams
    tokens = text.split()

    counts = Counter()

    for n in ngrams:
        for ngram in nltk.ngrams(tokens, n):
            word = " ".join(ngram)
            if word in sentinels:
                counts.update([word])

    return counts


def preprocess(text: str) -> str:
    """Preprocess the given text.

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
    text = "".join(filter(ALPHABET.__contains__, text.lower()))
    # Remove redundant whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

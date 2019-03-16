from nltk.corpus import stopwords as nltk_stop_words
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as sklearn_stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words


def stop_words():
    """Combine the list of stop words from Spacy, NLTK, and Sklearn."""
    words = [nltk_stop_words.words("english"), sklearn_stop_words, spacy_stop_words]
    return frozenset().union(*words)

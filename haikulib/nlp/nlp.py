import spacy
from nltk.corpus import stopwords as nltk_stop_words
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as sklearn_stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words

nlp = spacy.load("en", disable=["parser", "ner"])


def __stop_words():
    """Combine the list of stop words from Spacy, NLTK, and Sklearn."""
    return frozenset().union(nltk_stop_words.words("english"), sklearn_stop_words, spacy_stop_words)


STOPWORDS = __stop_words()


def remove_stopwords(text):
    """Remove stopwords from the given text."""
    return " ".join(word for word in text.split() if word not in STOPWORDS)


def lemmatize(texts):
    """Return the lemmatized version of the given text."""
    for doc in nlp.pipe(texts, batch_size=512, n_threads=-1):
        yield " ".join(token.lemma_ for token in doc)

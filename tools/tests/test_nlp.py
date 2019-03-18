import unittest

from nltk.corpus import stopwords as nltk_stop_words
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as sklearn_stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words

from tools.utils.nlp import STOPWORDS, lemmatize, remove_stopwords


class NlpTest(unittest.TestCase):
    def test_stopwords(self):
        nltk_set = set(nltk_stop_words.words("english"))
        sklearn_set = set(sklearn_stop_words)
        spacy_set = set(spacy_stop_words)

        diff1 = nltk_set - sklearn_set
        diff2 = nltk_set - spacy_set
        diff3 = sklearn_set - nltk_set
        diff4 = sklearn_set - spacy_set
        diff5 = spacy_set - nltk_set
        diff6 = spacy_set - sklearn_set

        differences = set.union(diff1, diff2, diff3, diff4, diff5, diff6)
        for diff in differences:
            self.assertIn(diff, STOPWORDS)

    def test_stopword_removal(self):
        stopwords = ["i", "am", "a", "but", "you", "are", "the"]
        for stopword in stopwords:
            self.assertIn(stopword, STOPWORDS)

        phrase = "i am a heathen but you are the worst"
        self.assertEqual(remove_stopwords(phrase), "heathen worst")

    def test_lemmatization(self):
        line1 = "she's my friend"
        line2 = "i'd rather not meet"
        line3 = "reading the stock futures"
        line4 = "the leaves still clinging"
        line5 = "beehives beneath kiwifruit vines"
        line6 = "warm winter rain/the beach and i/collecting sea glass"

        self.assertEqual(lemmatize(line1), "-PRON- be -PRON- friend")
        self.assertEqual(lemmatize(line2), "-PRON- would rather not meet")
        self.assertEqual(lemmatize(line3), "read the stock future")
        self.assertEqual(lemmatize(line4), "the leaf still cling")
        self.assertEqual(lemmatize(line5), "beehive beneath kiwifruit vine")
        # TODO: Why does "i'd" lemmatize to "-PRON- would" but "i" lemmatize to "i"?
        self.assertEqual(lemmatize(line6), "warm winter rain / the beach and i / collect sea glass")

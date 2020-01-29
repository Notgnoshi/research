import unittest

from haikulib.nlp import STOPWORDS, lemmatize, remove_stopwords


class NlpTest(unittest.TestCase):
    def test_stopword_removal(self):
        stopwords = ["i", "am", "a", "but", "you", "are", "the"]
        for stopword in stopwords:
            self.assertIn(stopword, STOPWORDS)

        phrase = "i am a heathen but you are the worst"
        self.assertEqual(remove_stopwords(phrase), "heathen worst")

    def test_lemmatization(self):
        lines = [
            "she's my friend",
            "i'd rather not meet",
            "reading the stock futures",
            "the leaves still clinging",
            "beehives beneath kiwifruit vines",
            "warm winter rain/the beach and i/collecting sea glass",
        ]
        expected_lemmas = [
            "-PRON- be -PRON- friend",
            "-PRON- would rather not meet",
            "read the stock future",
            "the leave still cling",
            "beehive beneath kiwifruit vine",
            # TODO: Why does "i'd" lemmatize to "-PRON- would" but "i" lemmatize to "i"?
            "warm winter rain / the beach and i / collect sea glass",
        ]

        lemmas = lemmatize(lines)
        for lemma, expected in zip(lemmas, expected_lemmas):
            self.assertEqual(lemma, expected)

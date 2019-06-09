import string
import unittest

from haikulib.utils.data import preprocess, tokenize


class TestPreprocess(unittest.TestCase):
    def test_casing(self):
        self.assertEqual(preprocess(string.ascii_lowercase), string.ascii_lowercase)
        self.assertEqual(preprocess(string.ascii_uppercase), string.ascii_lowercase)
        self.assertEqual(preprocess(string.ascii_letters), string.ascii_letters.lower())

    def test_spaces(self):
        nbsp = "on\xa0the\xa0playground"
        # 0x20 is ascii space.
        self.assertEqual(preprocess(nbsp), "on\x20the\x20playground")
        self.assertEqual(
            preprocess("one lip claps   tongue slipping"), "one lip claps tongue slipping"
        )

    def test_dashes(self):
        tshirt = "t-shirts out in force #"
        coffee = "fresh-ground french-pressed coffee"
        war = "bright clouds bleed a war–red"
        # NOTE: t-shirt *should* get parsed as tshirt.
        self.assertEqual(preprocess(tshirt), "t shirts out in force #")
        self.assertEqual(preprocess(coffee), "fresh ground french pressed coffee")
        self.assertEqual(preprocess(war), "bright clouds bleed a war red")

    def test_nonascii(self):
        russian = "это сущий разврат this is a veritable debauch"
        self.assertEqual(preprocess(russian), "this is a veritable debauch")

    def test_digits(self):
        self.assertEqual(preprocess(string.digits), string.digits)
        self.assertEqual(preprocess("abcd1234"), "abcd1234")

    def test_slashes(self):
        self.assertEqual(preprocess("one/two"), "one/two")

    def test_quotes(self):
        self.assertEqual(preprocess(r"'\""), "'")

    def test_punctuation(self):
        self.assertEqual(preprocess(string.punctuation), "#' /")

    def test_tokenize(self):
        haiku = "the day hides itself / within a grove of aspen / thunder pursues it #"
        word_tokens = haiku.split()
        char_tokens = list(haiku)

        self.assertSequenceEqual(word_tokens, tokenize(haiku, method="words"))
        self.assertSequenceEqual(char_tokens, tokenize(haiku, method="characters"))

        self.assertTrue(all(len(t) >= 1 for t in tokenize(haiku, method="words")))
        self.assertTrue(all(len(t) == 1 for t in tokenize(haiku, method="characters")))

        self.assertRaises(ValueError, tokenize, haiku, method="Words")
        self.assertRaises(ValueError, tokenize, haiku, method="asdfdf")

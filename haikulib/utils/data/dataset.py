"""Data representation(s) intended as model inputs as opposed to raw textual data."""
import functools
import itertools

import torch
from torch.utils.data import Dataset

from .data import get_df, tokenize


class TokenDictionary(dict):
    """A helper class to handle converting tokens to and from numeric indices."""

    def __init__(self, tokens):
        """Create a TokenDictionary from the provided tokens."""
        super().__init__()
        self.idx2token = []

        for token in tokens:
            if token not in self:
                self.idx2token.append(token)
                self[token] = len(self.idx2token) - 1

    def index(self, token):
        """Get the index corresponding to the provided token."""
        return self[token]

    def token(self, index):
        """Get the token corresponding to the provided index."""
        return self.idx2token[index]


class HaikuVocabIndexDataset(Dataset):
    """Each vocab item is represented as an integer index."""

    def __init__(self, seq_len=None, method="words"):
        """Create a vocab-index representation of the dataset.

        Each haiku will be broken into a number of overlapping sequences of tokens.
        A single X value is one such sequence. The corresponding Y value is the very
        next token in the sequence.

        Naturally, the length of the sequences should correspond to the number of tokens in the
        haiku in some manner. That is, sequences of characters should be longer than sequences of
        words.

        Further, sequences will *not* cross the boundary of two haiku.
        TODO: Experiment with sequences that *do* cross haiku boundaries.

        :param seq_len: The length of each sequence in tokens, defaults to 3 for 'words'
        tokenization, and defaults to 10 for 'characters' tokenization.
        :param method: One of 'words' or 'characters', defaults to 'words'

        TODO: Handle stopword removal or lemmatization.
        """
        tokenized_haiku = get_df()["haiku"].apply(functools.partial(tokenize, method=method))
        self.dictionary = TokenDictionary(itertools.chain.from_iterable(tokenized_haiku))

        # Map tokens to their indices before computing all of the sequences.
        tokenized_haiku = (
            (self.dictionary.index(token) for token in haiku) for haiku in tokenized_haiku
        )

        # Figure out a good default sequence length
        if seq_len is None and method == "words":
            seq_len = 3
        elif seq_len is None and method == "characters":
            seq_len = 10

        self.X, self.Y = self.get_sequences(tokenized_haiku, seq_len)

    def __len__(self):
        """Get the number of samples in this dataset."""
        return len(self.Y)

    def __getitem__(self, idx):
        """Get the sample at the provided index.

        Returns a (X, Y) pair.
        """
        return self.X[idx], self.Y[idx]

    @staticmethod
    def get_sequences(tokenized_haiku, seq_len):
        """Get the sequences and immediately following values from the haikus."""
        seqs = itertools.chain.from_iterable(
            map(
                HaikuVocabIndexDataset.get_sequences_single,
                tokenized_haiku,
                itertools.repeat(seq_len),
            )
        )

        # TODO: Consider preallocating these lists.
        X, Y = [], []
        for seq, target in seqs:
            X.append(seq)
            Y.append(target)

        return torch.as_tensor(X, dtype=torch.int32), torch.as_tensor(Y, dtype=torch.int32)

    @staticmethod
    def get_sequences_single(tokens, seq_len):
        """Return a sliding window generator yielding the sequences of the given haiku."""
        tokens = iter(tokens)
        seq = tuple(itertools.islice(tokens, seq_len + 1))

        # If the first element in the iterable is long enough, yield it.
        if len(seq) == seq_len + 1:
            yield seq[:-1], seq[-1]
        # Otherwise advance the sliding window down
        for token in tokens:
            seq = seq[1:] + (token,)
            yield seq[:-1], seq[-1]

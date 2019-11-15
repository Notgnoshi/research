---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: research
    language: python
    name: research
---

```
\author{Austin Gill}
\title{Exploratory Data Analysis -- Zipf's Law}
\maketitle
\tableofcontents
```

The goal of this notebook is to get comfortable with the Jupyter notebook workflow in the context of this project, as well as an opportunity to build some of the scaffolding around the haiku dataset.
There are a number of intricacies that have changed since I last used Jupyter, such as `nbconvert` attempting to use Inkscape to convert SVG images to PDFs before exporting the notebook to PDF!

The questions this notebook attempts to answer are

* Does Zipf's law hold for haiku?
* Does Zipf's law hold after removing stop words?
* Does Zipf's law hold after stemming/lemmatization?

```python
# Automagically reimport haikulib if it changes.
%load_ext autoreload
%autoreload 2

%config InlineBackend.figure_format = 'svg'
%matplotlib inline

from collections import Counter
import operator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

import spacy

from haikulib import data, utils, nlp
```

```python
_nlp = spacy.load("en", disable=["parser", "ner"])
pd.set_option('display.latex.repr', True)
pd.set_option('display.latex.longtable', True)
plt.rcParams["figure.figsize"] = (16 * 0.6, 9 * 0.6)

sns.set()
```

# Zipf's Law

Zipf's law states that the frequencies of words from a natural language corpus are inversely proportional to their rank in a frequency table. That is, a plot of their rank vs frequency on a log-log scale will be roughly linear.

For example, The first word in the table below is twice as frequent as the second word, and three times as frequent as the third.

| rank | value  | occurrences |
|------|--------|-------------|
| 1    | word 1 | 21          |
| 2    | word 2 | 10          |
| 3    | word 3 | 7           |

A plot of this frequency table on a log-log scale is shown below. Notice that the plot is roughly linear.

```python
ranks = np.array([1, 2, 3])
frequencies = np.array([21, 10, 7])

plt.plot(np.log(ranks), np.log(frequencies))
plt.plot(np.log(ranks), np.log(frequencies), ".")

plt.title("Example of Zipf's Law")
plt.xlabel("$\log(rank)$")
plt.ylabel("$\log(freq)$")
plt.show()
```

# Zipf's Law for our Dataset

One of the ways to represent a natural language corpus is with a bag-of-words representation, where all of the individual words of the corpus have been tossed in a container *without any surrounding context*.
This is a natural representation for the present work, as examining word frequencies does not require context.

For future work it will become necessary to represent words from the corpus as mathematical vectors.
This will allow us to use a more mathematical treatment of the problem, and allows us to (ab)use useful properties of vector spaces.
For example, there are pre-built models like [word2vec](https://jalammar.github.io/illustrated-word2vec/) or [GloVe](https://nlp.stanford.edu/projects/glove/) that would encode words like "man" and "woman" as vectors that are relatively close together, as opposed to, say, the vectors for "man" and "dog".
Further, these vector representations have the property that the vector between "man" and "woman" is close to the vector between "king" and "queen".

However, for the present work, a bag-of-words suffices.
So we get a bag-of-words representation of the dataset, and construct the frequency table.

```python
def get_freq_table(bag, thing="word"):
    """Get a frequency table representation of the given bag-of-<thing> representation."""
    assert isinstance(bag, Counter)
    things, frequencies = zip(*sorted(bag.items(), key=operator.itemgetter(1), reverse=True))
    things = np.array(things)
    frequencies = np.array(frequencies)
    ranks = np.arange(1, len(things) + 1)

    freq_table = pd.DataFrame({"rank": ranks, thing: things, "frequency": frequencies})
    return freq_table
```

The frequency table is just another view of the bag-of-words.
It contains no new information, but allows us to more easily examine the mathematical relationships of word frequencies.

```python
bag = data.get_bag_of(kind="words")
freq_table = get_freq_table(bag)
freq_table.head()
```

Plotting the ranks of each word vs their frequency on a log-log scale reveals that Zipf's law does seem to hold for most of the dataset.

```python
plt.plot(np.log(freq_table["rank"]), np.log(freq_table["frequency"]), '.', markersize=3)

plt.title("Haiku Word Frequency")
plt.xlabel("$\log(rank)$")
plt.ylabel("$\log(freq)$")
plt.show()
```

So then we find the words and their corresponding frequencies at the interesting points of the plot.

```python
def get_indices(df, column, values):
    """Gets the indices of values from the given column of the given dataframe."""
    indices = []
    for value in values:
        indices += df[column][df[column] == value].index.tolist()
    return indices
```

```python
indices = get_indices(freq_table, "word", ["the", "a", "of", "to", "i", "her", "his"])
interesting = freq_table.loc[indices]
interesting.head(7)
```

Unfortunately this, and much of the subsequent work involves a fair amount of manual tweaking.

```python
plt.plot(
    np.log(freq_table["rank"]), np.log(freq_table["frequency"]), ".", markersize=3
)

# This should be a crime.
x_adjust = np.array([0.1, -0.6, 0.15, -0.6, 0.2, -0.6, 0.0])
y_adjust = np.array([1.0, -1.2, 1.0, -1.3, 1.0, -1.3, 1.0])

for word, freq, rank, xa, ya in zip(
    interesting["word"],
    interesting["frequency"],
    interesting["rank"],
    x_adjust,
    y_adjust,
):
    plt.annotate(
        word,
        xy=(np.log(rank), np.log(freq) + ya / 20),
        xytext=(np.log(rank) + xa, np.log(freq) + ya),
        size=9,
        arrowprops={"arrowstyle": "-", "color": "k"},
    )

plt.title("Haiku Word Frequency")
plt.xlabel("$\log(rank)$")
plt.ylabel("$\log(freq)$")
plt.ylim((-0.5, 11.9))
# plt.savefig('zipfs-uncleaned.svg')
plt.show()
```

# Zipf's Law After Removing Stop Words

Unfortunately, the preceding diagram isn't very interesting.
That's because the most common words in natural language are often filler words with little individual meaning.
In the context of natural language processing, these are called **stop words**.

So we remove the stopwords from the bag of words and repeat our analysis.

```python
for stopword in nlp.STOPWORDS:
    if stopword in bag:
        del bag[stopword]

freq_table = get_freq_table(bag)
```

```python
plt.plot(
    np.log(freq_table["rank"]), np.log(freq_table["frequency"]), ".", markersize=3
)

plt.title("Haiku Word Frequency")
plt.xlabel("$\log(rank)$")
plt.ylabel("$\log(freq)$")
plt.show()
```

The plot retains the same characteristics, albeit with a slightly less linear shape.

After removing the stop words, the most frequent words start to show characteristics unique to haiku.

```python
freq_table.head(15)
```

```python
indices = get_indices(freq_table, "word", ["moon", "rain", "day", "night", "snow", "winter", "summer", "spring", "autumn"])

interesting = freq_table.loc[indices]
```

```python
plt.plot(
    np.log(freq_table["rank"]), np.log(freq_table["frequency"]), ".", markersize=3
)

# This should also be a crime.
x_adjust = np.array([-0.35, -0.9, -0.23, -0.9, -0.1, -0.7, 0.3, -0.7, 0.4])
y_adjust = np.array([1.0, -1.0, 1.1, -1.1, 1.1, -1.4, 1.0, -1.45, 1.0])

for word, freq, rank, xa, ya in zip(
    interesting["word"],
    interesting["frequency"],
    interesting["rank"],
    x_adjust,
    y_adjust,
):
    plt.annotate(
        word,
        xy=(np.log(rank), np.log(freq) + ya / 20),
        xytext=(np.log(rank) + xa, np.log(freq) + ya),
        size=8,
        arrowprops={"arrowstyle": "-", "color": "k"},
    )

plt.title("Haiku Word Frequency")
plt.xlabel("$\log(rank)$")
plt.ylabel("$\log(freq)$")
plt.xlim((-0.5, 10.5))
plt.ylim((-0.5, 9))
# plt.savefig("zipfs-cleaned.svg")
plt.show()
```

In the context of Zipf's law, this diagram isn't very revealing.
But as exploratory data analysis undertaken to understand the haiku dataset, it is quite illuminating.

We can immediately tell that weather and seasons are major themes in haiku.


# Word Frequencies After Stemming/Lemmatization

There are two computational approaches for getting the root form of a word - stemming and lemmatization.

Stemming involves a sequence of rules used to strip off suffixes of the word to reduce it to its stem - which notably might not be a word.
For example, "leaves" and "leaving" might both be stemmed to form "leav".
Further, because stemming operates by removing parts of the word, it would fail to stem "better" and "good" the same.

Notably, stemming is unaware of the vocabulary.
It is a purely algorithmic process of applying grammatical rules to remove prefixes and suffixes.

Lemmatization on the other hand, is aware of vocabulary.
It is a more sophisticated technique that returns the word to its base dictionary form via morphological analysis.
Lemmatization is much more costly than stemming, and is often performed using a machine learning model.

```python
bag = data.get_bag_of(kind='words')

for stopword in nlp.STOPWORDS:
    if stopword in bag:
        del bag[stopword]

feq_table = get_freq_table(bag)
```

## Stemming

There are many approaches to stemming words, but the most common approaches are the Porter, Lancaster, and Snowball stemmers.

So in order to get a feel for how stemming effects, we will build a bag-of-stems for each of the above stemmers.

```python
# Build a new bag of stems
porter_stems = Counter()
lancaster_stems = Counter()
snowball_stems = Counter()

porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
snowball_stemmer = SnowballStemmer("english")
```

```python
for word, frequency in zip(freq_table["word"], freq_table["frequency"]):
    stem = porter_stemmer.stem(word)
    if stem in porter_stems:
        porter_stems[stem] += frequency
    else:
        porter_stems[stem] = frequency

    stem = lancaster_stemmer.stem(word)
    if stem in lancaster_stems:
        lancaster_stems[stem] += frequency
    else:
        lancaster_stems[stem] = frequency

    stem = snowball_stemmer.stem(word)
    if stem in snowball_stems:
        snowball_stems[stem] += frequency
    else:
        snowball_stems[stem] = frequency
```

Each of the stemmers produce similar results.

```python
print("Original: length:", len(bag), "common words:", bag.most_common(15), "\n\n",)
print(
    "Porter: length:",
    len(porter_stems),
    "common stems:",
    porter_stems.most_common(15), "\n\n",
)
print(
    "Lancaster: length:",
    len(lancaster_stems),
    "common stems:",
    lancaster_stems.most_common(15), "\n\n",
)
print(
    "Snowball: length:",
    len(snowball_stems),
    "common stems:",
    snowball_stems.most_common(15), "\n\n",
)
```

We see the largest compression from the Lancaster stemmer.
So we use the Lancaster stems to plot the same frequency curve as before.

```python
freq_table = get_freq_table(lancaster_stems)
plt.plot(
    np.log(freq_table["rank"]), np.log(freq_table["frequency"]), ".", markersize=3
)

plt.title("Haiku Stem Frequency")
plt.xlabel("$\log(rank)$")
plt.ylabel("$\log(freq)$")
plt.show()
```

The shape of the curve does not appear to have changed much from the frequency plot with the stop words removed, except slightly more curved.
Perhaps there just aren't that many variants of each word.
Or perhaps Zipf's law holds on natural language word stems as well as the words themselves.
I think that is more likely.


## Lemmatization

Lemmatization is a more involved process, and takes quite a bit more time.

There are two lemmatizers that I will use: one from NLTK, and one from SpaCy.
However, the NLTK WordNet lemmatizer supports two modes: with, and without Part-Of-Speech (POS) tagging.

So we procede with the three variants.

```python
freq_table = get_freq_table(bag)

wn_lemmas = Counter()
wn_pos_lemmas = Counter()
spacy_lemmas = Counter()
```

```python
lem = WordNetLemmatizer()
for word, frequency in zip(freq_table["word"], freq_table["frequency"]):
    lemma = lem.lemmatize(word)
    if lemma in wn_lemmas:
        wn_lemmas[lemma] += frequency
    else:
        wn_lemmas[lemma] = frequency
```

```python
def get_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tags = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    # Default to a noun if the POS is unknown.
    return tags.get(tag, wordnet.NOUN)
```

```python
for word, frequency in zip(freq_table["word"], freq_table["frequency"]):
    lemma = lem.lemmatize(word, get_pos(word))
    if lemma in wn_pos_lemmas:
        wn_pos_lemmas[lemma] += frequency
    else:
        wn_pos_lemmas[lemma] = frequency
```

Note that SpaCy lemmatizes tokens in the provided corpus as a part of its model construction, so we build a full model for each of the words in the dataset.
This is *not* what SpaCy was designed for, but should we lemmatize the entire dataset, we would not be able associate the lemma's frequency with that of the original word.
This is because the bag-of-words is already compressed - there are no duplicate tokens - only an annotation of how frequent each token is.

```python
%time
# horrendously slow
for word, frequency in zip(freq_table["word"], freq_table["frequency"]):
    # This is not what SpaCy was meant for.
    doc = _nlp(word)
    token = doc[0]
    lemma = token.lemma_

    if lemma in spacy_lemmas:
        spacy_lemmas[lemma] += frequency
    else:
        spacy_lemmas[lemma] = frequency
```

```python
print("original: length:", len(bag), "most common:", bag.most_common(15), "\n\n",)
print(
    "WordNet: length:",
    len(wn_lemmas),
    "most common:",
    wn_lemmas.most_common(15), "\n\n",
)
print(
    "WordNet with POS: length:",
    len(wn_pos_lemmas),
    "most common:",
    wn_pos_lemmas.most_common(15), "\n\n",
)
print(
    "spaCy: length:",
    len(spacy_lemmas),
    "most common:",
    spacy_lemmas.most_common(15), "\n\n",
)
```

Note that each of the lemmatizers identifies the same most common lemmas, but with different frequencies.
The SpaCy lemmatizer does the most compression, so plot the same frequency curve as before using the SpaCy lemmas.

```python
freq_table = get_freq_table(spacy_lemmas)

plt.plot(
    np.log(freq_table["rank"]),
    np.log(freq_table["frequency"]),
    ".",
    markersize=3,
)

plt.title("Haiku Lemma Frequency")
plt.xlabel("$\log(rank)$")
plt.ylabel("$\log(freq)$")
plt.show()
```

The pattern is the same as before.


# Conclusion

My conclusion is that Zipf's law does in fact hold for haiku.
The initial thought was that it might not because haiku are a compressed form of natural language.

Interestingly, it holds before and after removing stop words - words like "an" and "the", which are quite common.
Zipf's law is stated abstractly for tokens in a natural language, but holds even for the stems and lemmas of those tokens.
This makes sense, and is not surprising.

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #raw -->
\author{Austin Gill}
\title{Exploratory Data Analysis -- Zipf's Law}
\maketitle
\tableofcontents
<!-- #endraw -->

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

%config InlineBackend.figure_format = 'retina'
%matplotlib inline

import operator
from collections import Counter
from urllib.request import urlopen

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns

from haikulib import data, nlp, utils
```

```python
# _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
# pd.set_option("display.latex.repr", True)
# pd.set_option("display.latex.longtable", True)
pd.set_option("display.max_colwidth", None)
plt.rcParams["figure.figsize"] = (16 * 0.8, 9 * 0.8)

sns.set(style="whitegrid")
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

# plt.title("Example of Zipf's Law")
plt.xlabel("$\log(rank)$")
plt.ylabel("$\log(freq)$")
plt.savefig("contrived.png", dpi=240)
plt.show()
```

# Compare Haiku to Ambrose Bierce's Writing

Before we proceed with looking at Zipf's law for haiku, we should have a baseline to compare it to.
I'm a sucker for the dark, cynicism of Ambrose Bierce, so let's download his works from Project Gutenberg and examine how they look with respect to Zipf's law.

```python
part1 = "https://www.gutenberg.org/cache/epub/13541/pg13541.txt"
part2 = "https://www.gutenberg.org/cache/epub/13334/pg13334.txt"

part1 = urlopen(part1).read().decode("utf-8")
part2 = urlopen(part2).read().decode("utf-8")
```

```python
def get_freq_table(bag):
    """Get a frequency table representation of the given bag-of-words representation."""
    assert isinstance(bag, Counter)
    words, frequencies = zip(
        *sorted(bag.items(), key=operator.itemgetter(1), reverse=True)
    )
    words = np.array(words)
    frequencies = np.array(frequencies)
    ranks = np.arange(1, len(words) + 1)

    freq_table = pd.DataFrame({"rank": ranks, "word": words, "frequency": frequencies})
    return freq_table
```

```python
corpus = " ".join(part1.split()) + " ".join(part2.split())
tokens = [nlp.preprocess(t) for t in corpus.split()]
bag = Counter(tokens)
freq_table_bierce = get_freq_table(bag)
freq_table_bierce.head()
```

```python
plt.plot(
    np.log(freq_table_bierce["rank"]),
    np.log(freq_table_bierce["frequency"]),
    ".",
    markersize=3,
)

plt.xlabel("$\log(rank)$")
plt.ylabel("$\log(freq)$")
plt.savefig("bierce.png", dpi=240)
plt.show()
```

```python
for stopword in nlp.STOPWORDS:
    if stopword in bag:
        del bag[stopword]

stop_table_bierce = get_freq_table(bag)

plt.plot(
    np.log(freq_table_bierce["rank"]),
    np.log(freq_table_bierce["frequency"]),
    ".",
    markersize=3,
    label="bierce with stopwords",
)
plt.plot(
    np.log(stop_table_bierce["rank"]),
    np.log(stop_table_bierce["frequency"]),
    ".",
    markersize=3,
    label="bierce without stopwords",
)

plt.xlabel("$\log(rank)$")
plt.ylabel("$\log(freq)$")
plt.legend()
plt.savefig("bierce-nostopwords.png", dpi=240)
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


The frequency table is just another view of the bag-of-words.
It contains no new information, but allows us to more easily examine the mathematical relationships of word frequencies.

```python
bag = data.get_bag_of(kind="words", add_tags=False)
del bag["/"]
freq_table = get_freq_table(bag)
freq_table.head()
```

```python
plt.plot(
    np.log(freq_table["rank"]),
    np.log(freq_table["frequency"]),
    ".",
    markersize=3,
    label="haiku with stopwords",
)
plt.plot(
    np.log(freq_table_bierce["rank"]),
    np.log(freq_table_bierce["frequency"]),
    ".",
    markersize=3,
    label="bierce with stopwords",
)

plt.xlabel("$\log(rank)$")
plt.ylabel("$\log(freq)$")
plt.legend()
plt.savefig("haiku.png", dpi=240)
plt.show()
```

Plotting the ranks of each word vs their frequency on a log-log scale reveals that Zipf's law does seem to hold for most of the dataset.


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
plt.plot(np.log(freq_table["rank"]), np.log(freq_table["frequency"]), ".", markersize=3)

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

# plt.title("Haiku Word Frequency")
plt.xlabel("$\log(rank)$")
plt.ylabel("$\log(freq)$")
plt.ylim((-0.5, 11.9))
plt.savefig("haiku-annotated.png", dpi=240)
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

stop_table = get_freq_table(bag)
```

```python
plt.plot(
    np.log(freq_table["rank"]),
    np.log(freq_table["frequency"]),
    ".",
    markersize=3,
    label="haiku with stopwords",
)
plt.plot(
    np.log(stop_table["rank"]),
    np.log(stop_table["frequency"]),
    ".",
    markersize=3,
    label="haiku without stopwords",
)

# plt.title("Haiku Word Frequency")
plt.xlabel("$\log(rank)$")
plt.ylabel("$\log(freq)$")
plt.legend()
plt.savefig("haiku-nostopwords.png", dpi=240)
plt.show()
```

The plot retains the same characteristics, albeit with a slightly less linear shape.

After removing the stop words, the most frequent words start to show characteristics unique to haiku.

```python
stop_table.head(15)
```

```python
indices = get_indices(
    stop_table,
    "word",
    ["moon", "rain", "day", "night", "snow", "winter", "summer", "spring", "autumn"],
)

interesting = stop_table.loc[indices]
```

```python
plt.plot(np.log(stop_table["rank"]), np.log(stop_table["frequency"]), ".", markersize=3)

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

plt.xlabel("$\log(rank)$")
plt.ylabel("$\log(freq)$")
plt.xlim((-0.5, 10.5))
plt.ylim((-0.5, 9.2))
plt.savefig("haiku-no-stopwords-annotated.png", dpi=240)
plt.show()
```

# Conclusion

My conclusion is that Zipf's law does in fact hold for haiku.
The initial thought was that it might not because haiku are a compressed form of natural language.

Interestingly, it holds before and after removing stop words - words like "an" and "the", which are quite common.
Zipf's law is stated abstractly for tokens in a natural language, but holds even for the stems and lemmas of those tokens.
This makes sense, and is not surprising.

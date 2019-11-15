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
\title{Exploratory Data Analysis -- The Syllabic Structure of Haiku}
\maketitle
\tableofcontents
```

# The Haiku Form

Haiku are a form of Japanese poetry.
They are the first three lines of a *waka*, which is a 5-7-5-7-7 poem, also known as a *tanka* or *uta*.

*Waka* were often strung together into *renga*, which is a linked verse dialog between multiple poets.
One poet would compose the first 5-7-5 syllables, then another would respond with the next 7-7 syllables, and so on.

Of these long poems, the first three 5-7-5 lines are called a *hokku*.
The *hokku* are the most recognizable and impressionable verses of *renga*, analgous to the first verse and chorus of a popular song.
*Renga* were composed in meetings of several poets, and it was common for poets to come prepared with pre-composed *hokku*.

Note that the strict 17-syllable structure is difficult (impossible) to maintain in translation.
Thus many translators strive to match the style, theme, and mood of a Japanese haiku, maintaining its three-line structure, but abandoning the strict syllabic requirement.
Further note that, as with any art style, there was experimentation with freer verse styles, in both content and structure.

However in *The Haiku Form*, Joan Giroux defines

> A haiku is a 17-syllable poem arranged in three lines of 5, 7, and 5 syllables, having some reference to the season and expressing the poet's union with nature.

and still suggests that the 17-syllable structure applies to native English haiku just as well.

The purpose of this notebook is to empirically answer the question:
How many haiku actually follow the common 5-7-5 syllable pattern?

> asshole questioning
>
> doesn't know about haiku
>
> 5-7-5 bitch

by examining the haiku corpus scraped from popular haiku submission websites.

```python
# Automagically reimport haikulib if it changes.
%load_ext autoreload
%autoreload 2

%config InlineBackend.figure_format = 'svg'
%matplotlib inline

import collections
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

from haikulib import data
```

```python
experiment_dir = data.get_data_dir() / "experiments" / "eda" / "syllables"
experiment_dir.mkdir(parents=True, exist_ok=True)

pd.set_option("display.latex.repr", True)
pd.set_option("display.latex.longtable", True)
pd.set_option('display.max_colwidth', -1)

# plt.rcParams has no effect if in the same cell as the matplotlib import.
# See: https://github.com/ipython/ipython/issues/11098
plt.rcParams["figure.figsize"] = (16 * 0.6, 9 * 0.6)
sns.set()

df = data.get_df()
df.head()
```

# Total Syllable Counts

First, we look at the distribution of lines in our corpus.
The vast majority of haiku in the corpus are composed of three lines, with a few outliers on either side.

In order to make the syllable count analysis easier, we consider only the three-line haiku.

```python
print(collections.Counter(df["lines"]))
# Consider only those haiku that consist of three lines.
df = df[df["lines"] == 3]
# Reindex, so that adding a syllable count column isn't borked.
df.reset_index(inplace=True, drop=True)
```

As expected, the distribution of the total number of syllables is roughly normal.

```python
sns.distplot(
    df["total_syllables"],
    bins=np.arange(5, 25),
    kde_kws={"bw": 0.4},
    hist_kws={"align": "left"},
)
plt.title("Haiku total syllable count")
plt.xlabel("syllables")
plt.ylabel("density")
plt.show()
```

However, I expected the distribution to be centered on seventeen syllables, as that's the traditional structure discussed above.
The actual center is thirteen syllables.

The five-number summary of the total syllable count agrees.

```python
df["total_syllables"].describe()
```

The extreme outliers on either side are interesting, so we examine them to see if we need to prune the dataset.

```python
df[df["total_syllables"] <= 4]
```

```python
df[df["total_syllables"] >= 26]
```

The outliers on both sides seem subjectively reasonable, if strict adherence to the traditional seventeen-syllable structure is abandonded.
They each 

**Note:** This outlier analysis revealed the presence of the following zero-syllable haiku in the corpus:

> ♡ ♡ ♡ ♡ ♡
>
> ♡ ♡ ♡ ♡ ♡ ♡ ♡
>
> ♡ ♡ ♡ ♡ ♡

This was treated as zero-syllables because the dataset preprocessing step ensures that the haiku are converted to lowercase ASCII-encoded alphabetic characters (`a` through `z`).
Thus the above "haiku" was converted to the literal string `"/ / #"` (the `/` symbols are line separators, and `#` marks the end of the haiku).

This "haiku" was removed from the dataset, and the outlier analysis re-ran (since these notebooks are intended to be reproducible).

# Syllable Counts per Line

Then we look at the syllable count for each line in the haiku corpus.

```python
one, two, three = zip(*df["syllables"])

bins = np.arange(1, 10)
# Using the bandwidth kde kwarg to produce a smooth estimated kernel
# that doesn't spike with every bin.
kde_kws = {"bw": 0.4}
hist_kws = {"align": "left"}


sns.distplot(
    one,
    label="first",
    bins=bins,
    kde_kws=kde_kws,
    hist_kws=hist_kws,
)
sns.distplot(
    two,
    label="second",
    bins=bins,
    kde_kws=kde_kws,
    hist_kws=hist_kws,
)
sns.distplot(
    three,
    label="third",
    bins=bins,
    kde_kws=kde_kws,
    hist_kws=hist_kws,
)

plt.title("Haiku syllables per line")
plt.legend()
plt.xlabel("syllables")
plt.ylabel("density")
plt.show()
```

We can see that there is a clear distinction between the distributions of the middle and surrounding lines.
This agrees with my expectations, but it is surprising to find that the middle distribution is centered on five, not seven, syllables.
It's also interesting to note that the distributions of the first and last lines are similar, yet the distribution of the third line's syllables is slightly skewed higher.


# Common Syllable Structures

Again, restricted to three-line haiku, we can look at the most common syllabic structures occurring in the corpus.

```python
counts = collections.Counter(df["syllables"])
total = sum(counts.values())

rows = {
    "syllables": list(counts.keys()),
    "count": list(counts.values()),
    "proportion": [v / total for v in counts.values()],
}

syllables = pd.DataFrame(rows)
syllables.sort_values(by="count", inplace=True, ascending=False)
syllables.reset_index(inplace=True, drop=True)
syllables.head(10)
```

We see that the 5-7-5 structure *is* the most common, but that *it occurs in only 2.8% of the corpus*.
This is surprising.
I had expected the traditional form to be dominant over the others, with a few outliers.

```python
plt.plot(np.log(syllables["count"]))

plt.title("Distribution of syllabic structures in haiku")
plt.ylabel("$\log(freq)$")
plt.xlabel("$rank$")
plt.show()
```

With the exception of the most common structures, the distribution of syllabic structures in the haiku corpus is exponential with respect to rank.
Note that the stair-step nature of the bottom end of the distribution is due to the discrete nature of the frequencies.
There are a number of haiku with unique syllabic structures (all with the same frequency), and there are a number of *pairs* of haiku with the same structure, and so on.

# Conclusion

Empirically, the syllabic structure of haiku (restricted to my corpus) is far more varied than expected.
Previous analysis on the content and vocabulary (with respect to word frequencies and color content) met my expectations, and confirmed the stereotypical seasonal content of haiku.
This analysis, however, indicates that the one distinguishing feature of haiku in popular media --- its syllabic structure --- is not nearly as distinctive in the actual corpus.

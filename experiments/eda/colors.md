---
jupyter:
  jupytext:
    formats: ipynb,md
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
\title{Exploratory Data Analysis -- Color}
\maketitle
\tableofcontents
```

The goal of this notebook is to build a color palette of my haiku dataset in the same vein as a PyCon 2017 conference talk titled [Gothic Colors: Using Python to understand color in nineteenth century literature](https://www.youtube.com/watch?v=3dDtACSYVx0).

This conference talk was the first application of programming to a soft science that I recall being exposed to, and it's made a lasting impression.
Ever since watching the talk, I've wanted to apply scientific techniques to solve non-scientific, soft, and natural problems.

Here, I intend to parse the use of color from the haiku in an intelligent manner -- one that is aware that the word "rose" has different meanings in the sentences

* "I picked a rose."
* "Her shoes were rose colored."
* "He rose to greet me."

In a sense, however, the first two uses both contribute to the sense of a "color palette" for haiku, so we care only about excluding the third case.

In order to do perform this differentiation, the haiku corpus must be part-of-speech tagged.
That is, each word must be annotated with its part of speech.
This is a daunting task for such a large corpus -- as of the time of this notebook, the corpus contains over 178,000 words!

Fortunately POS-tagging is not a new problem, and there exist out-of-the-box methods for performing POS tagging.

```python
# Automagically reimport haikulib if it changes.
%load_ext autoreload
%autoreload 2

%config InlineBackend.figure_format = 'svg'
%matplotlib inline

import collections
import colorsys
import itertools

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import webcolors
from IPython.display import Image

import haikulib.eda.colors
from haikulib import data, nlp, utils

data_dir = data.get_data_dir() / "experiments" / "eda" / "colors"
data_dir.mkdir(parents=True, exist_ok=True)
pd.set_option("display.latex.repr", True)
pd.set_option("display.latex.longtable", True)
plt.rcParams["figure.figsize"] = (16, 9)
sns.set()
```

# The Naive Approach

It's often useful to implement a simpler version of a feature before implementing the full functionality.
So before performing POS-tagging and more intelligent color identification, we simply look for any occurance of a color name in the haiku corpus.

We do so by stripping the `/` and `#` meta-tokens from each haiku, then look for any $n$ -grams from the corpus that match our list of color names.
We use $n \in \{1, 2, 3\}$.

```python
# Form list of haiku without '/' and '#' symbols
df = data.get_df()
corpus = []

for haiku in df["haiku"]:
    corpus.append(" ".join(line.strip(" #") for line in haiku.split("/")))

color_names = {
    r["color"]: r["hex"] for _, r in haikulib.eda.colors.get_colors().iterrows()
}
```

```python
%%time
naive_colors = collections.Counter()
for haiku in corpus:
    # Update the color counts for this haiku.
    naive_colors.update(nlp.count_tokens_from(haiku, color_names, ngrams=[1, 2, 3]))
```

Here, we build a data frame of the color occurences for ease of use in visualization.
Before it was sufficient to use the `collections.Counter()` object directly in generating the word cloud, but now we prefer more a more structured data form.

```python
naive_color_counts = pd.DataFrame(
    {
        "color": list(naive_colors.keys()),
        "count": list(naive_colors.values()),
        "hex": [color_names[c] for c in naive_colors],
    }
)

total_color_count = sum(row["count"] for index, row in naive_color_counts.iterrows())

print(f"There are {total_color_count} occurences of color in the corpus")
print(f"There are {len(naive_color_counts)} unique colors")

naive_color_counts.head(10)
```

# Parsing Colors using Part-Of-Speech Tagging


Rather than implement the color parsing as a part of this notebook, it is performed as a part of the `haikulib.eda` library so that the color parsing can be done *on creation* of the `haiku.csv` cleaned data file.
This enables using the results of this analysis in other exploration.

However, it's useful to examine the implementation of the color parsing code to demonstrate how it works.
In order to do this in a manner that prevents copy-pasting implementations --- which inevitably leads to multiple out-of-sync versions of the same code --- I wrote a small introspective helper function to render the source code of the given function as syntax-highlighted HTML in a Jupyter notebook.

```python
utils.display_source("haikulib.utils", "display_source")
```

We can determine if a word is a color simply by checking if it is contained in our master list of colors, and by checking if it is an adjective or a noun.

```python
utils.display_source("haikulib.eda.colors", "is_color")
```

However, this relies on each word in the corpus being tagged with their corresponding part-of-speech.
This too is simple.

```python
utils.display_source("haikulib.nlp", "pos_tag")
```

Notice that the line separators and end-of-haiku symbols are ignored, as they do not have a part of speech.

Now we can simply find all of the colors in a given haiku as follows.

```python
# Modified to test colors of all three sizes.
haiku = "dark blue lines / in a light olive green sea salt / dreams #"
haiku_colors = [
    tagged_word[0]
    for tagged_word in nlp.pos_tag(haiku)
    if haikulib.eda.colors.is_color(tagged_word)
]
print(haiku_colors)
```

But what about finding the color "dark blue"?
In order to find multi-word colors, we need to parse and test $n$ -grams from the haiku.

```python
utils.display_source("haikulib.eda.colors", "find_colors")
```

Notice that we only use the `is_color()` method discussed above to determine if single-token words are colors.
The requirements for ngrams being a color is relaxed to a simple containment check --- is the ngram in our list of known colors?

Further notice that there is soul-crushing logic used to parse the colors `["light olive green", "sea"]` from the string `"light olive green sea"` instead of the colors `["olive", "green", "sea", "olive green", "light olive green"]`.

```python
haikulib.eda.colors.find_colors(nlp.pos_tag(haiku))
```

Then we can parse colors from the haiku before saving the haiku in the `haiku.csv` data file.
This enables spatial exploration of the colors, because they are associated with individual haiku rather than building a simple `collections.Counter` object of colors as above with the naive approach.

```python
utils.display_source("haikulib.data.initialization", "init_csv")
```

```python
df = data.get_df()
df.tail(6)
```

We can also produce a `DataFrame` containing the colors, their counts, and their HTML color codes as above.

```python
pos_tagging_color_counts = haikulib.eda.colors.get_colors()

total_color_count = pos_tagging_color_counts["count"].sum()
used_color_count = pos_tagging_color_counts["count"].astype(bool).sum(axis=0)

print(f"There are {total_color_count} occurences of color in the corpus")
print(f"There are {used_color_count} unique colors")

pos_tagging_color_counts[["color", "count", "hex"]].head(10)
```

Compare the POS-tagging results with those from the naive approach, summarized again below.
Notice that we pruned over twenty unique colors by using POS-tagging, and pruned over *three thousand* occurences of color words that were not tagged as adjectives or nouns, or duplicated by the occurence of an ngram.

```python
total_color_count = naive_color_counts["count"].sum()

print(f"There are {total_color_count} occurences of color in the corpus")
print(f"There are {len(naive_color_counts)} unique colors")

naive_color_counts.head(10)
```

# Color Palette Visualization

There are a number of palette visualization techniques we could use.
We will visualize the haiku color palette using
* Word cloud
* Histogram
* Pie Chart
* Ordered Grid
* Spectrum


## Word Cloud

One visualization technique to understand the usage of color is to use a word cloud, as discussed in another notebook.
An advantage of this technique is that it readily displays not only the colors, but the color names as well.
Additionally, it gives a good sense for the frequency of each color.

Unfortunately, a word cloud does not give a sense for the overall color palette, since the representation is unstructured and random, putting disparate colors next to each other.

```python
Image(data_dir / ".." / "word_clouds" / "colors.png")
```

## Histogram

The color histogram, sorted by frequency, is shown below.

```python
colors = haikulib.eda.colors.get_colors()
colors.sort_values(by=["hsv", "count"], ascending=False, inplace=True)
used_colors = colors.loc[colors["count"] != 0].copy()
used_colors.sort_values(by="count", ascending=False, inplace=True)
```

```python
_ = plt.bar(
    range(len(used_colors)),
    used_colors["count"],
    color=used_colors["rgb"],
    width=1,
    linewidth=0,
    log=True,
)
plt.savefig(data_dir / "histogram.eps")
plt.show()
```

However, the are other ways we might display the same information.

```python
used_colors.sort_values(by="hsv", ascending=False, inplace=True)
_ = plt.bar(
    range(len(used_colors)),
    used_colors["count"],
    color=used_colors["rgb"],
    width=1,
    linewidth=0,
    log=True,
)
plt.savefig(data_dir / "histogram-spectrum.eps")
plt.show()
```

```python
background = plt.bar(
    range(len(colors)),
    height=12 ** 3,
    width=1,
    linewidth=0,
    color=colors["rgb"],
    log=True,
    alpha=0.8,
)
foreground = plt.bar(
    range(len(colors)),
    height=colors["count"],
    width=3,
    linewidth=0,
    color="black",
    log=True,
)
plt.savefig(data_dir / "histogram-spectrum-background.pdf")
plt.show()
```

## Polar Histogram

We have several options for a polar histogram.
We can

* Sort the colors radially, by their hue or their frequency
* Use fixed or proportional radii
* Use fixed or proportional wedge widths
* Use a fixed or proportional division of $[0, 2\pi]$ for the wedge's angular locations

```python
def pairwise_difference(seq):
    for l, r in utils.pairwise(seq):
        yield r - l
    # Loop back around to the front.
    yield 2 * np.pi - seq[-1]


def accumulate(seq):
    _sum = 0
    for s in seq:
        yield _sum
        _sum += s
```

### Sorted By Frequency

First we'll sort the colors by their frequency and display them radially.

```python
used_colors.sort_values(by="count", ascending=False, inplace=True)

ax = plt.subplot(111, projection="polar")

thetas = 2 * np.pi * used_colors["count"] / used_colors["count"].sum()
thetas = np.array(list(accumulate(thetas)))
widths = np.array(list(pairwise_difference(thetas)))
radii = np.log(used_colors["count"])

_ = ax.bar(
    x=thetas,
    height=radii,
    width=widths,
    color=used_colors["rgb"],
    linewidth=0,
    align="edge",
)
plt.savefig(data_dir / "count-proportional-theta-radii-width.eps")
plt.show()
```

```python
ax = plt.subplot(111, projection="polar")

_ = ax.bar(
    x=thetas,
    # Plot the same information with a fixed height.
    height=1,
    width=widths,
    color=used_colors["rgb"],
    linewidth=0,
    align="edge",
)
plt.savefig(data_dir / "count-proportional-theta-width-fixed-height.eps")
plt.show()
```

### Sorting by Hue

However, the goal with the color visualization is to get a sense of the color palette in general, so it's useful to arrange similar colors next to each other.

Unfortunately this is a hard problem in general, because colors are three dimensional, and we are attempting to sort them in a single dimension!
There are techniques that utilize Hilbert curves to sort colors in two dimensions, but for our immediate work sorting by hue is sufficient.

```python
# Do a lexicographical sort by hue, then saturation, then value.
used_colors.sort_values(by="hsv", ascending=False, inplace=True)
```

First, we use a fixed angular offset, accompanied by fixed width bars.

```python
ax = plt.subplot(111, projection="polar")

thetas = np.linspace(0, 2 * np.pi, len(used_colors), endpoint=False)
widths = 4 * np.pi / len(used_colors)
radii = np.log(used_colors["count"])

_ = ax.bar(
    x=thetas,
    height=radii,
    width=widths,
    color=used_colors["rgb"],
    linewidth=0,
    align="edge",
)
plt.savefig(data_dir / "hue-proportional-radii-fixed-theta-width.eps")
plt.show()
```

```python
ax = plt.subplot(111, projection="polar")

thetas = 2 * np.pi * used_colors["count"] / used_colors["count"].sum()
thetas = np.array(list(accumulate(thetas)))
widths = np.array(list(pairwise_difference(thetas)))
radii = np.log(used_colors["count"])

_ = ax.bar(
    x=thetas,
    height=1,
    width=widths,
    color=used_colors["rgb"],
    linewidth=0,
    align="edge",
)
plt.savefig(data_dir / "hue-proportional-theta-width-fixed-radii.eps")
plt.show()
```

```python
ax = plt.subplot(111, projection="polar")

_ = ax.bar(
    x=thetas,
    height=radii,
    width=widths,
    color=used_colors["rgb"],
    linewidth=0,
    align="edge",
)
plt.savefig(data_dir / "hue-proportional-theta-radii-width.eps")
plt.show()
```

## Chronological Grid


## Color Adjacency Graph

**TODO: Maybe move to a different notebook?**

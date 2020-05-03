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
\title{Data Analysis -- Haiku Word Similarity}
\maketitle
\tableofcontents
<!-- #endraw -->

See the gensim [tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_annoy.html) on Word2Vec similarity queries.

The purpose of this notebook is to use a more advanced technique for computing haiku similarity queries than those discussed in the word adjacency graph notebook.
Computing document similarity is a complex task that has received much treatment in the NLP literature.

See

* [Cosine Similarity for Vector Space Models (iii)](http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/)
* [Text Similarities: Estimate the degree of similarity between two texts](https://medium.com/@adriensieg/text-similarities-da019229c894) and corresponding [https://github.com/adsieg/text_similarity](https://github.com/adsieg/text_similarity) and [https://github.com/nlptown/nlp-notebooks](https://github.com/nlptown/nlp-notebooks)
* [Comparing Sentence Similarity Methods](https://nlp.town/blog/sentence-similarity/)
* [Quick Review on Text Clustering and Text Similarity Approaches](http://www.lumenai.fr/blog/quick-review-on-text-clustering-and-text-similarity-approaches)

```python
# Automagically reimport haikulib if it changes.
%load_ext autoreload
%autoreload 2

%config InlineBackend.figure_format = 'svg'
%matplotlib inline

import collections
import itertools

import gensim
import gensim.downloader as gs_downloader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.similarities.index import AnnoyIndexer

from haikulib import data, nlp, utils
```

```python
experiment_dir = data.get_data_dir() / "experiments" / "similarity"
experiment_dir.mkdir(parents=True, exist_ok=True)
pd.set_option("display.latex.repr", True)
pd.set_option("display.latex.longtable", True)
plt.rcParams["figure.figsize"] = (16 * 0.6, 9 * 0.6)
sns.set()
```

# Using pretrained Word2Vec with Annoy

We use the pretrained Word2Vec model from the Google News dataset.
This dataset is ~2GB (3 million words).

If our haiku corpus was much larger, we should not use a model trained on new articles.
Instead, we should train the Word2Vec model from scratch on the haiku corpus, or at least fine-tune an existing pretrained model.

```python
%%time
# Run jupterlab with --NotebookApp.iopub_msg_rate_limit=1.0e10
# to silence the IO rate limiting warnings that the progress bar causes.
# Note that downloading the dataset takes some time, but loading it into
# memory takes 1:47...
word2vec = gs_downloader.load("word2vec-google-news-300")
```

[Annoy](https://github.com/spotify/annoy) (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings to search for points in space that are close to a given query point. It also creates large read-only file-based data structures that are mmapped into memory so that many processes may share the same data.

It works by using random projections and by building up a tree. At every intermediate node in the tree, a random hyperplane is chosen, which divides the space into two subspaces. This hyperplane is chosen by sampling two points from the subset and taking the hyperplane equidistant from them.

This is done $k$ times so that we get a forest of trees. $k$ has to be tuned to your need, by looking at what tradeoff you have between precision and performance.

```python
%%time
# Use 50 trees.
annoy_index = AnnoyIndexer(word2vec, 50)
```

```python
query = word2vec.wv["cherry"]
# Query for most similar words to 'cherry'.
# Note that this Word2Vec model is pre-trained! It knows nothing about haiku!
annoy = word2vec.wv.most_similar([query], topn=10, indexer=annoy_index)

for word, sim in annoy:
    print(f"word: {word},\tsimilarity: {sim}")
```

# Training a Word2Vec model on the haiku corpus


# Training a fastText model on the haiku corpus

The GenSim documentation for [fastText](https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html) seem to indicate that it can work better than Word2Vec for small datasets (which we have)


# Word2Vec and fastText comparison

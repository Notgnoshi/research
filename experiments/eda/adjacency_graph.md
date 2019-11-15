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

# Word Adjacency Graph

What words are adjacent to each other?

```python
%load_ext autoreload
%autoreload 2

%config InlineBackend.figure_format = 'svg'
%matplotlib inline

import collections
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import spacy

from haikulib import data, utils, nlp
```

```python
_nlp = spacy.load("en", disable=["parser", "ner"])
# If the parser and NER are disabled, it's safe to increase the length limit
_nlp.max_length = 2_500_000

sns.set()

DATA_DIR = data.get_data_dir() / "experiments" / "generation" / "adjacency_graph"
DATA_DIR.mkdir(exist_ok=True)
```

```python
def adjacency_graph(corpus):
    corpus = corpus.split("#")
    edges = collections.Counter()

    for haiku in corpus:
        haiku = haiku.split()
        haiku = (w for w in haiku if w != "'s" and w != "'" and w != "/")
        edges.update(utils.pairwise(haiku))

    graph = nx.DiGraph()
    for edge, weight in edges.items():
        graph.add_edge(*edge, weight=weight)

    return graph
```

```python
df = data.get_df()

doc = _nlp(" ".join(nlp.remove_stopwords(h) for h in df["haiku"])[:500])
lemmatized_corpus = " ".join(token.lemma_ for token in doc)
graph = adjacency_graph(lemmatized_corpus)

nx.write_gexf(graph, str(DATA_DIR / "partial-adjacencies.gexf"))
nx.draw(graph, with_labels=True)
```

```python
doc = _nlp(" ".join(nlp.remove_stopwords(h) for h in df["haiku"]))
lemmatized_corpus = " ".join(token.lemma_ for token in doc)
graph = adjacency_graph(lemmatized_corpus)

nx.write_gexf(graph, str(DATA_DIR / "full-adjacencies.gexf"))
```

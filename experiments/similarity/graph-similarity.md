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
\title{Data Analysis -- Haiku Graph Similarity}
\maketitle
\tableofcontents
```

In order to get an intuitive feeling for how well (how creative) a haiku generator is, we compare the similarity of generated haiku with each of the haiku in the training corpus.

```python
# Automagically reimport haikulib if it changes.
%load_ext autoreload
%autoreload 2

%config InlineBackend.figure_format = 'svg'
%matplotlib inline

import collections
import itertools

import grakel
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import seaborn as sns

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

# Word Level Adjacency Graph Similarity

We lemmatize the haiku so that the nodes in the adjacency graphs carry more meaning.

```python
%%time
corpus = data.get_df()
# Exceedingly slow
corpus["lemma"] = list(nlp.lemmatize(corpus["haiku"]))
generated = data.get_generated_df()
generated["lemma"] = list(nlp.lemmatize(generated["haiku"]))
generated.head()
```

Then we convert the haiku into their adjacency graphs.
NetworkX graphs are used only to draw the generated graphs, all computation is done on the GraKeL graphs.

```python
def haiku2edges(haiku):
    edges = collections.Counter()
    tokens = nltk.word_tokenize(haiku)
    edges.update(utils.pairwise(tokens))
    return edges


def edges2grakel(edges):
    tokens = set(itertools.chain.from_iterable(edges))
    # Even though the node objects are the same as the node labels,
    # we still need the labels for some unknown reason.
    return grakel.Graph(edges, node_labels={k: k for k in tokens})


def edges2nx(edges):
    graph = nx.DiGraph()
    for edge, weight in edges.items():
        graph.add_edge(*edge, weight=weight)
    return graph
```

```python
%%time
gen_edges = [haiku2edges(h) for h in generated["lemma"]]
gen_graphs = [edges2grakel(e) for e in gen_edges]
gen_nx_graphs = [edges2nx(e) for e in gen_edges]

corpus_edges = [haiku2edges(h) for h in corpus["lemma"]]
corpus_graphs = [edges2grakel(e) for e in corpus_edges]

# There are enough haiku in the dataset that we need to use sparse representations.
graph_kernel = grakel.kernels.WeisfeilerLehman(
    n_iter=2, normalize=True, base_kernel=(grakel.kernels.VertexHistogram, {"sparse": True})
)
```

Here's what the adjacency graph looks like for one of the generated haiku.
Notice that for this particular haiku, there are no tokens (aside from `/`) that have multiple incoming or outgoing edges.
In other words, the adjacency graph is a linear sequence of words, so we expect the graph similarity measures to pick out haiku that share as long of a common subsequence as possible.

```python
graph = gen_nx_graphs[1]
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_size=900)

plt.show()
```

```python
%%time
for query_graph, query in zip(gen_graphs, generated["haiku"]):
    graph_kernel.fit([query_graph])
    kernel = graph_kernel.transform(corpus_graphs)

    # number of similar haiku to find
    n = 3
    indices = np.argsort(kernel[:, 0])[-n:]
    similar = corpus.iloc[indices]
    print("query:", query)
    for sim in similar["haiku"]:
        print("\tsimilar:", sim)
```

Our intuition about the graph similarity measures is correct, for the particular queries given, it seems to be measuring similarity based on the length of common subsequences.

# Character Level Adjacency Graph Similarity

```python
def haiku2edges(haiku):
    edges = collections.Counter()
    # Iterate over the pairs of characters in the haiku string.
    edges.update(utils.pairwise(haiku))
    return edges
```

```python
%%time
gen_edges = [haiku2edges(h) for h in generated["lemma"]]
gen_graphs = [edges2grakel(e) for e in gen_edges]
gen_nx_graphs = [edges2nx(e) for e in gen_edges]

corpus_edges = [haiku2edges(h) for h in corpus["lemma"]]
corpus_graphs = [edges2grakel(e) for e in corpus_edges]

# There are enough haiku in the dataset that we need to use sparse representations.
graph_kernel = grakel.kernels.WeisfeilerLehman(
    n_iter=2, normalize=True, base_kernel=(grakel.kernels.VertexHistogram, {"sparse": True})
)
```

It's much harder to extract visual meaning from the character level adjacency graphs.
However, perhaps the graph similarity measures can extract more meaning, since there are more nodes in the graphs, and the graphs are no longer mostly linear sequences.

```python
graph = gen_nx_graphs[1]
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_size=900)

plt.show()
```

```python
%%time
for query_graph, query in zip(gen_graphs, generated["haiku"]):
    graph_kernel.fit([query_graph])
    kernel = graph_kernel.transform(corpus_graphs)

    # number of similar haiku to find
    n = 8
    indices = np.argsort(kernel[:, 0])[-n:]
    similar = corpus.iloc[indices]
    print("query:", query)
    for sim in similar["haiku"]:
        print("\tsimilar:", sim)
```

The word level adjacency graphs seem to base similarity off of haiku sharing common subsequences.
The character level similarity measures seem to base similarity only on sharing a common word.
Conclusion: it's not sufficient to use adjacency graphs to compute haiku similarity.

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
import numpy as np
import nltk
import networkx as nx
import pandas as pd
import seaborn as sns

from haikulib import data, nlp, utils
```

```python
data_dir = data.get_data_dir() / "experiments" / "similarity"
data_dir.mkdir(parents=True, exist_ok=True)
pd.set_option("display.latex.repr", True)
pd.set_option("display.latex.longtable", True)
plt.rcParams["figure.figsize"] = (16 * 0.6, 9 * 0.6)
sns.set()
```

```python
def get_generated_df():
    return pd.read_csv(
        # TODO: Actually generate this CSV file.
        data.get_data_dir() / "experiments" / "generation" / "knesser-ney" / "generated.csv",
        index_col=0,
    )
```

```python
%%time
corpus = data.get_df()
# Exceedingly slow
corpus["lemma"] = list(nlp.lemmatize(corpus["haiku"]))
generated = get_generated_df()
generated["lemma"] = list(nlp.lemmatize(generated["haiku"]))
generated.head()
```

```python
def haiku2edges(haiku):
    edges = collections.Counter()
    tokens = nltk.word_tokenize(haiku)
    edges.update(utils.pairwise(tokens))
    return edges

def edges2grakel(edges):
    tokens = set(itertools.chain.from_iterable(edges))
    return grakel.Graph(edges, node_labels={k:k for k in tokens})

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

corpus_edges = (haiku2edges(h) for h in corpus["lemma"])
corpus_graphs = [edges2grakel(e) for e in corpus_edges]

# There are enough haiku in the dataset that we need to use sparse representations.
graph_kernel = grakel.kernels.WeisfeilerLehman(n_iter=2, normalize=True, base_kernel=(grakel.kernels.VertexHistogram, {"sparse": True}))
```

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

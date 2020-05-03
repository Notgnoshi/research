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

# Topic Modeling

```python
%config InlineBackend.figure_format = 'svg'
%matplotlib inline

import logging
import sys
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.corpora import Dictionary
from gensim.models import LdaModel, Phrases
from nltk.stem.wordnet import WordNetLemmatizer

from haikulib.data import get_df
from haikulib.nlp import remove_stopwords

# Building the LdaModel freezes Firefox due to so much frigging logger output.
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.CRITICAL,
    stream=sys.stdout,
)

# Print pandas.DataFrame's nicely.
pd.set_option("display.latex.repr", True)
pd.set_option("display.latex.longtable", True)
# Use the 16x9 aspect ratio, with a decent size.
plt.rcParams["figure.figsize"] = (16 * 0.6, 9 * 0.6)
# Use a better default matplotlib theme.
sns.set()
```

```python
df = get_df()
documents = (remove_stopwords(haiku).split() for haiku in df["haiku"])
documents = ([token for token in document if len(token) > 1] for document in documents)

wnl = WordNetLemmatizer()
documents = [[wnl.lemmatize(token) for token in document] for document in documents]

bigrams = Phrases(documents, min_count=10)
for document in documents:
    for token in bigrams[document]:
        if "_" in token:
            document.append(token)
```

```python
dictionary = Dictionary(documents)
# Filter out vocab words that appear in fewer than 20 haiku, and those that occur in more than 50%
dictionary.filter_extremes(no_below=20, no_above=0.5)
corpus = [dictionary.doc2bow(document) for document in documents]

print("Unique tokens:", len(dictionary))
print("Documents:    ", len(corpus))
```

```python
num_topics = 30
chunksize = 2000
# The number of training epochs
passes = 50
# The number of loops over each document
iterations = 600
eval_every = None

_ = dictionary[0]
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha="auto",
    eta="auto",
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every,
)
```

```python
topics = model.top_topics(corpus, topn=10)
avg_topic_coherence = sum(t[1] for t in topics) / num_topics
print("Average topic coherence:", avg_topic_coherence)

pprint(topics)
```

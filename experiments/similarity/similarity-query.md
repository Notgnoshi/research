# Similarity Queries

Given a haiku (generated or not), find the most similar haiku from the training corpus.
This should give a partial measure of how "creative" the generative model is.

```python
%config InlineBackend.figure_format = 'svg'
%matplotlib inline

import logging
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity
from nltk.stem.wordnet import WordNetLemmatizer

from haikulib.data import get_data_dir, get_df
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
wnl = WordNetLemmatizer()
df["lemmas"] = df["haiku"].apply(
    lambda haiku: " ".join(wnl.lemmatize(token) for token in haiku.split())
)

df["nostopwords"] = df["lemmas"].apply(remove_stopwords)

df.tail()
```

```python
%%time

# Need to tokenize each haiku and filter out the `/` separators.
documents = (haiku for haiku in df["nostopwords"])
documents = [[token for token in haiku.split() if token != "/"] for haiku in documents]
dictionary = Dictionary(documents)
dictionary.filter_extremes(no_below=5, no_above=0.95)
corpus = [dictionary.doc2bow(haiku) for haiku in documents]

print("Unique tokens:", len(dictionary))
print("Documents:    ", len(corpus))

dimensions = 500
model = LsiModel(corpus, id2word=dictionary, num_topics=500)
index = MatrixSimilarity(model[corpus])

index_path = str(get_data_dir() / "similarity.index")
index.save(index_path)
index = MatrixSimilarity.load(index_path)
```

```python
query = "spoonfuls / of medication / and loneliness"
# query = "cherry blossom / a white moth / in my hand"
# query = "summer / all these extra prayers / at the dead puppies"

tokens = query.split()
tokens = [wnl.lemmatize(token) for token in tokens]
query_bow = dictionary.doc2bow(tokens)
query_lsi = model[query_bow]

# Find the cosine distance between the query and all of the training documents.
similarities = index[query_lsi]
# Find the top N most similar indices
N = 10
topn = similarities.argsort()[-N:][::-1]

print("query:", query)
print()
for i in topn:
    print(f"index: {i}   \tsim: {similarities[i]}      \thaiku: {df.iloc[i]['haiku']}")
```

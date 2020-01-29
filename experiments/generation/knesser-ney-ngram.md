---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
# Automagically reimport haikulib if it changes.
%load_ext autoreload
%autoreload 2

%config InlineBackend.figure_format = 'svg'
%matplotlib inline

import random
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns

from haikulib import data, nlp, utils
```

```python
data_dir = data.get_data_dir() / "experiments" / "generation" / "knesser-ney"
data_dir.mkdir(parents=True, exist_ok=True)
pd.set_option("display.latex.repr", True)
pd.set_option("display.latex.longtable", True)
plt.rcParams["figure.figsize"] = (16, 9)
sns.set()
```

```python
order=3
haiku = data.get_df()
words = data.get_bag_of(kind="words")

tokens = nltk.word_tokenize(" ".join(haiku["haiku"]))
ngrams = nltk.everygrams(tokens, max_len=order)

lm = nltk.lm.models.KneserNeyInterpolated(order=order)
lm.fit([ngrams], vocabulary_text=tokens)
```

```python
next_word = None
# TODO: Pick a random seed word.
# TODO: Consider using start-of-haiku symbols like '<h>'...'</h>'.
haiku = ["summer"]
seed = random.randint(0, 2**32 - 1)
print(f"Generating with seed: {seed}")

while next_word != "</h>":
    next_word = lm.generate(random_seed=seed, text_seed=haiku) if len(haiku) < 50 else "</h>"
    haiku.append(next_word)

print(haiku)
```

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

```python
%config InlineBackend.figure_format = 'svg'
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import scipy as sp
import seaborn as sns
```

```python
sns.set()
plt.rcParams["figure.figsize"] = (16 * 0.7, 9 * 0.7)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
```

```python
def softmax(values, temperature=1.0):
    preds = np.exp(values / temperature)
    return preds / np.sum(preds)

def sample(values, temperature=1.0):
    preds = np.exp(np.log(values) / temperature)
    return preds / np.sum(preds)
```

```python
x = np.linspace(start=0, stop=1, num=500)
# relative likelihoods
likelihoods = sp.stats.argus.pdf(x, chi=1, loc=0, scale=1)
# Add random noise
likelihoods1 = likelihoods + np.random.normal(loc=0, scale=0.005, size=len(likelihoods))
likelihoods2 = likelihoods + np.random.normal(loc=0, scale=0.005, size=len(likelihoods))
softs = softmax(likelihoods1)

plt.plot(x, softs, label=r"$\mathrm{softmax}(\vec x)$")
plt.plot(x, softmax(likelihoods2, temperature=0.8), label=r"$\mathrm{softmax}(\vec x, t=0.8)$")
plt.plot(x, sample(softs, temperature=0.8), label=r"$\mathrm{sample}(\mathrm{softmax}(\vec x), t=0.8)$")
# plt.plot(x, sample(softs, temperature=0.6), label=r"$sample(softmax(\vec x), t=0.6)$")

plt.title("Softmax and temperature sampling")
plt.legend()

plt.show()
```

```python
x = np.linspace(start=0, stop=1, num=100)
# relative likelihoods
likelihoods = sp.stats.argus.pdf(x, chi=1, loc=0, scale=1)
# Add random noise
likelihoods = likelihoods + np.random.normal(loc=0, scale=0.005, size=len(likelihoods))

plt.plot(x, softmax(likelihoods, temperature=1.0), label=r"$\mathrm{softmax}(\vec x)$")
plt.plot(x, softmax(likelihoods, temperature=1.5), label=r"$\mathrm{softmax}(\vec x, t=1.5)$")
plt.plot(x, softmax(likelihoods, temperature=2), label=r"$\mathrm{softmax}(\vec x, t=2.0)$")
plt.plot(x, softmax(likelihoods, temperature=0.8), label=r"$\mathrm{softmax}(\vec x, t=0.8)$")
plt.plot(x, softmax(likelihoods, temperature=0.6), label=r"$\mathrm{softmax}(\vec x, t=0.6)$")

plt.title("Softmax and temperature sampling")
plt.legend()
plt.tick_params(labelbottom=False)
# tikzplotlib.save("softmax-temperature.tikz", figurewidth=r"\textwidth")
plt.show()
```

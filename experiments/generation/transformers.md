---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import logging

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# logging.basicConfig(level=logging.INFO)
```

```python
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
text = "Who was Jim Hanson ? Jim Hanson was a"
indexed_tokens = tokenizer.encode(text)
tokens = torch.tensor([indexed_tokens])
```

```python
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

tokens = tokens.to("cuda")
model.to("cuda")

with torch.no_grad():
    outputs = model(tokens)
    predictions = outputs[0]

predicted_index = torch.argmax(predictions[0, -1:]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
print(predicted_text)
```

#!/usr/bin/env python3
import os
import random
import sys

# Make the datasets module importable.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import DATASETS
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"

set_session(tf.Session(config=config))

dataset = DATASETS["nietzsche"]

if not dataset.exists():
    dataset.download()

with open(dataset.filename) as f:
    corpus = f.read().lower()

print("corpus length:", len(corpus))

# TODO: Use word tokens (including punctuation) rather than characters
tokens = sorted(list(set(corpus)))
print("number of tokens:", len(tokens))

# Map tokens to indices and vice versa
token_indices = {t: i for i, t in enumerate(tokens)}
index_tokens = {i: t for i, t in enumerate(tokens)}

# Cut the corpus into overlapping sequences of tokens, and the token at the end of each sequence.
seqlen = 40
step = 3
sequences = []
next_tokens = []

for i in range(0, len(corpus) - seqlen, step):
    sequences.append(corpus[i : i + seqlen])
    next_tokens.append(corpus[i + seqlen])
print("number of sequences:", len(sequences))

# Vectorize the sequences and next tokens into a one-hot encoding
x = np.zeros((len(sequences), seqlen, len(tokens)), dtype=np.bool)
y = np.zeros((len(sequences), len(tokens)), dtype=np.bool)
for i, sequence in enumerate(sequences):
    for j, token in enumerate(sequence):
        x[i, j, token_indices[token]] = True
    y[i, token_indices[next_tokens[i]]] = True

print("x:", sys.getsizeof(x) / (1000 * 1000), "MB")
print("y:", sys.getsizeof(y) / (1000 * 1000), "MB")

model = Sequential()
model.add(LSTM(128, input_shape=(seqlen, len(tokens))))
model.add(Dense(len(tokens), activation="softmax"))

optimizer = RMSprop(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)


def sample(predictions, temperature=1.0):
    predictions = np.asarray(predictions, dtype=np.float64)
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    # Return the prediction with the highest probability
    return np.argmax(probabilities)


def cb_epoch_end(epoch, _):
    print("=" * 80)
    print("Generating text after epoch:", epoch)
    # Pick a random seed from the corpus
    start = random.randint(0, len(corpus) - seqlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("diversity:", diversity)
        generated = ""
        seed = corpus[start : start + seqlen]
        print("seed:", seed)
        sys.stdout.write(generated)

        # Generate 400 tokens starting with the given seed
        for _ in range(400):
            x_pred = np.zeros((1, seqlen, len(tokens)), dtype=np.bool)
            for j, token in enumerate(seed):
                x_pred[0, j, token_indices[token]] = True

            predictions = model.predict(x_pred, verbose=0)[0]
            next_token = sample(predictions, diversity)
            next_token = index_tokens[next_token]

            generated += next_token
            # TODO: Prefer deque over list for this operation
            seed = seed[1:] + next_token

            sys.stdout.write(next_token)
            sys.stdout.flush()
        print()
        print("-" * 80)


callback = LambdaCallback(on_epoch_end=cb_epoch_end)

model.fit(x, y, batch_size=128, epochs=60, callbacks=[callback])

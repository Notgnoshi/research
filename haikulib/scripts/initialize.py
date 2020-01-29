#!/usr/bin/env python3
"""Prepare the haiku CSV dataset."""
from haikulib.data import get_df
from haikulib.data.initialization import init_csv

if __name__ == "__main__":
    # Downloading the NLTK data is done in the Docker image.
    # init_nltk()
    init_csv()
    print(get_df().tail())

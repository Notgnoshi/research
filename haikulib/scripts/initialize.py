#!/usr/bin/env python3
"""Prepare the haiku CSV dataset."""
import pathlib
import sys

REPO_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = REPO_DIR / "data"

# Add repository root directory to path so that haikulib.utils.data is importable.
sys.path.append(str(REPO_DIR))

from haikulib.data import get_df
from haikulib.data.initialization import init_csv

if __name__ == "__main__":
    init_csv()
    df = get_df()
    print(df.tail())

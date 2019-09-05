#!/usr/bin/env python3
from haikulib.data import get_df

if __name__ == "__main__":
    df = get_df(init=True)
    print(df.tail())

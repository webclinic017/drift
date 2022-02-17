from hashlib import sha256
import pandas as pd


def hash_df(df: pd.DataFrame) -> str:
    s = str(df.columns) + str(df.index) + str(df.values)
    return sha256(s.encode()).hexdigest()


def hash_series(df: pd.Series) -> str:
    s = str(df.name) + str(df.index) + str(df.values)
    return sha256(s.encode()).hexdigest()

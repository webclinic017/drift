import pandas as pd
import numpy as np
import os
import string
import random
from itertools import dropwhile


def get_files_from_dir(path: str) -> list[str]:
    return [
        f
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and not f.startswith(".")
    ]


def get_first_valid_return_index(series: pd.Series) -> int:
    double_nested_results = np.where(
        np.logical_and(series != 0, np.logical_not(pd.isna(series)))
    )
    if len(double_nested_results) == 0:
        return 0
    nested_result = double_nested_results[0]
    if len(nested_result) == 0:
        return 0
    return nested_result[0]


def get_last_non_na_index(series: pd.Series, index: int) -> int:
    return next(
        dropwhile(lambda x: pd.isna(x[1]), enumerate(reversed(series[: index + 1])))
    )[0]


def flatten(list_of_lists: list) -> list:
    return [item for sublist in list_of_lists for item in sublist]


def weighted_average(df: pd.DataFrame, weights_source: str) -> pd.Series:
    if df.shape[1] == 0:
        return df
    mean_df = df.iloc[:, 0]
    weights = df.loc[weights_source]

    for i, row in df.iterrows():
        if i == weights_source:
            continue
        mean_df.loc[i] = (row * weights).sum() / df.loc[weights_source].sum()

    return mean_df


def drop_columns_if_exist(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df = df.drop(column, axis=1)
    return df


def random_string(n: int) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def equal_except_nan(row: pd.Series):
    if np.isnan(row.iloc[0]) or np.isnan(row.iloc[1]):
        return np.nan
    if row.iloc[0] == row.iloc[1]:
        return 1.0
    else:
        return 0.0


def drop_until_first_valid_index(
    df: pd.DataFrame, series: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    first_valid_index = max(
        get_first_valid_return_index(df.iloc[:, 0]),
        get_first_valid_return_index(series),
    )
    return df.iloc[first_valid_index:], series.iloc[first_valid_index:]

from typing import Literal
import pandas as pd
import numpy as np

def get_first_valid_return_index(series: pd.Series) -> int:
    double_nested_results = np.where(np.logical_and(series != 0, np.logical_not(np.isnan(series))))
    if len(double_nested_results) == 0:
        return 0
    nested_result = double_nested_results[0]
    if len(nested_result) == 0:
        return 0
    return nested_result[0]

def flatten(list_of_lists: list) -> list:
    return [item for sublist in list_of_lists for item in sublist]

def weighted_average(df: pd.DataFrame, weights_source: str) -> pd.DataFrame:
    mean_df = df.iloc[:,0]
    weights = df.loc[weights_source]

    for i, row in df.iterrows():
        if i == weights_source: continue
        mean_df.loc[i] = (row * weights).sum() / df.loc[weights_source].sum()

    return mean_df


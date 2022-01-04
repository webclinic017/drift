from fracdiff.sklearn import FracdiffStat
import pandas as pd
import numpy as np
from feature_extractors.utils import apply_log_if_necessary_series

def feature_fractional_differentiation(df: pd.DataFrame, period: int, is_log_return: bool) -> pd.Series:
    frac_diff = FracdiffStat(window = period)
    input_series = df["close"].to_numpy().reshape(-1, 1)
    result = frac_diff.fit_transform(input_series)
    return pd.Series(result.squeeze(), index = df.index)

def feature_fractional_differentiation_log(df: pd.DataFrame, period: int, is_log_return: bool) -> pd.Series:
    series = feature_fractional_differentiation(df, period, is_log_return)
    return apply_log_if_necessary_series(series, "fracdiff")
    
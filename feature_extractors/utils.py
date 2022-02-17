import pandas as pd
import numpy as np
from scipy.stats import shapiro


def get_close_low_high(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = df["close"]
    low = df["low"]
    high = df["high"]
    return close, low, high


def apply_log_if_necessary_series(series: pd.Series, name: str) -> pd.Series:
    values = series.to_numpy()
    no_of_unique_values = np.unique(values)
    if len(no_of_unique_values) < 4:
        return series
    is_normal = shapiro(values).pvalue > 0.05
    if not is_normal:
        # print("Applying log to column: " + column)
        min_value = np.min(series)
        series = (series + min_value).apply(lambda x: np.log(x))
        is_normal_after_log = shapiro(series).pvalue > 0.05
        if not is_normal_after_log:
            print("Failed to normalize column: ", name)
    return series

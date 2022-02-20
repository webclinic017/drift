import pandas as pd
import numpy as np
from scipy.stats import shapiro


def get_close_low_high(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = df["close"]
    low = df["low"]
    high = df["high"]
    return close, low, high

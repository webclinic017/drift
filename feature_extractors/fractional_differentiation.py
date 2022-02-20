from fracdiff.sklearn import FracdiffStat
import pandas as pd
import numpy as np


def feature_fractional_differentiation(df: pd.DataFrame, period: int) -> pd.Series:
    frac_diff = FracdiffStat(window=period)
    input_series = df["close"].to_numpy().reshape(-1, 1)
    result = frac_diff.fit_transform(input_series)
    return pd.Series(result.squeeze(), index=df.index)

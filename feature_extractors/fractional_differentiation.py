from fracdiff.sklearn import FracdiffStat
import pandas as pd
import numpy as np

def feature_fractional_differentiation(df: pd.DataFrame, period: int, is_log_return: bool) -> pd.Series:
    feature_selector = FracdiffStat(window = period)
    input_series = df["close"].to_numpy().reshape(-1, 1)
    feature_selector.fit(input_series)
    result = feature_selector.transform(input_series)
    return pd.Series(np.log(result.squeeze()), index = df.index)
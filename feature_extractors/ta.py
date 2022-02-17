import pandas_ta as ta
import pandas as pd
from feature_extractors.utils import get_close_low_high


def feature_EBSW(df: pd.DataFrame, period: int) -> pd.Series:
    close, low, high = get_close_low_high(df)

    return ta.ebsw(close, period)

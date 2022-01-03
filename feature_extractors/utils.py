import pandas as pd

def get_close_low_high(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = df['close']
    low = df['low']
    high = df['high']
    return close, low, high
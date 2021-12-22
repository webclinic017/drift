import pandas as pd
import numpy as np

## Utility functions

def __get_close_low_high(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = df['close']
    low = df['low']
    high = df['high']
    return close, low, high

## Feature extractors

def feature_debug_future_lookahead(df: pd.DataFrame, period: int, is_log_return: bool) -> pd.Series:
    return df['returns'].shift(-period)

def feature_lag(df: pd.DataFrame, period: int, is_log_return: bool) -> pd.Series:
    assert period > 0
    return df['returns'].shift(period)

def feature_day_of_week(df: pd.DataFrame, period: int, is_log_return: bool) -> pd.DataFrame:
    return pd.get_dummies(pd.DatetimeIndex(df.index).dayofweek, drop_first=True, prefix="date_day_week").set_index(df.index)

def feature_day_of_month(df: pd.DataFrame, period: int, is_log_return: bool) -> pd.DataFrame:
    return pd.get_dummies(pd.DatetimeIndex(df.index).day, drop_first=True, prefix="date_day_month").set_index(df.index)

def feature_month(df: pd.DataFrame, period: int, is_log_return: bool) -> pd.DataFrame:
    return pd.get_dummies(pd.DatetimeIndex(df.index).month, drop_first=True, prefix="date_month").set_index(df.index)

def feature_vol(df: pd.DataFrame, period: int, is_log_return: bool) -> pd.Series:
    return df['returns'].rolling(period).std() * (252**0.5)

def feature_mom(df: pd.DataFrame, period: int, is_log_return: bool) -> pd.Series:
    if is_log_return:
        return np.log(df['close']).diff(period)
    else:
        return df['close'].pct_change(period)

def feature_STOK(df: pd.DataFrame, period: int, is_log_return: bool) -> pd.Series:
    close, low, high = __get_close_low_high(df)

    STOK = ((close - low.rolling(period).min()) / (high.rolling(period).max() - low.rolling(period).min())) * 100
    return STOK

def feature_STOD(df: pd.DataFrame, period: int, is_log_return: bool) -> pd.Series:
    stok = feature_STOK(df, period, is_log_return)
    return stok.rolling(3).mean()

def feature_RSI(df: pd.DataFrame, period: int, is_log_return: bool) -> pd.Series:
    returns = df['returns']
    delta = returns.diff().dropna()
    u=delta*0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean() / \
    d.ewm(com=period-1, adjust=False).mean()
    return 100-100/(1+rs)

def feature_ROC(df: pd.DataFrame, period: int, is_log_return: bool) -> pd.Series:
    returns = df['returns']
    M = returns.diff(period - 1)
    N = returns.shift(period - 1)
    return pd.Series(((M / N) * 100), name = 'ROC_' + str(period))
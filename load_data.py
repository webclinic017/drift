#%%
import pandas as pd
import os
import numpy as np
from utils.technical_indicators import ROC, RSI, STOK, STOD

#%%

def load_files(path: str, add_features: bool, log_returns: bool, narrow_format: bool = False) -> pd.DataFrame:
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and not f.startswith('.')]
    dfs = [__load_df(os.path.join(path,f), f.split('.')[0], add_features, log_returns, narrow_format) for f in files]
    if narrow_format:
        dfs = pd.concat(dfs, axis=0).fillna(0.)
    else:
        dfs = pd.concat(dfs, axis=1).fillna(0.)

    dfs.index = pd.DatetimeIndex(dfs.index)

    if add_features:
        dfs['day_month'] = dfs.index.day
        dfs['day_week'] = dfs.index.dayofweek
        dfs['month'] = dfs.index.month

    if narrow_format:
        return dfs
    else:
        return dfs.drop(index=dfs.index[0], axis=0)

def __load_df(path: str, prefix: str, add_features: bool, log_returns: bool, narrow_format: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, index_col=0).fillna(0)

    if log_returns:
        df['returns'] = np.log(df['close']).diff(1)
    else:
        df['returns'] = df['close'].pct_change()

    if add_features:
        # volatility (10, 20, 30 days)
        df['vol_10'] = df['returns'].rolling(10).std()*(252**0.5)
        df['vol_20'] = df['returns'].rolling(20).std()*(252**0.5)
        df['vol_30'] = df['returns'].rolling(30).std()*(252**0.5)
        df['vol_60'] = df['returns'].rolling(30).std()*(252**0.5)

        # momentum (10, 20, 30, 60, 90 days)
        if log_returns:
            df['mom_10'] = np.log(df['close']).diff(10)
            df['mom_20'] = np.log(df['close']).diff(20)
            df['mom_30'] = np.log(df['close']).diff(30)
            df['mom_60'] = np.log(df['close']).diff(60)
            df['mom_90'] = np.log(df['close']).diff(90)
        else:
            df['mom_10'] = df['close'].pct_change(10)
            df['mom_20'] = df['close'].pct_change(20)
            df['mom_30'] = df['close'].pct_change(30)
            df['mom_60'] = df['close'].pct_change(60)
            df['mom_90'] = df['close'].pct_change(90)


    df['roc_10'] = ROC(df['close'], 10)
    df['roc_30'] = ROC(df['close'], 30)

    df['rsi_10'] = RSI(df['close'], 10)
    df['rsi_30'] = RSI(df['close'], 30)
    df['rsi_100'] = RSI(df['close'], 30)

    df['stok_10'] = STOK(df['close'], df['low'], df['high'], 10)
    df['stod_10'] = STOD(df['close'], df['low'], df['high'], 10)
    df['stok_30'] = STOK(df['close'], df['low'], df['high'], 30)
    df['stod_30'] = STOD(df['close'], df['low'], df['high'], 30)
    df['stok_200'] = STOK(df['close'], df['low'], df['high'], 200)
    df['stod_200'] = STOD(df['close'], df['low'], df['high'], 200)


    df = df.replace([np.inf, -np.inf], 0.)
    df = df.drop(columns=['open', 'high', 'low', 'close'])
    if narrow_format:
        df["ticker"] = np.repeat(prefix, df.shape[0])
    else: 
        df.columns = [prefix + "_" + c for c in df.columns]
    return df

# %%
def create_target_cum_forward_returns(df: pd.DataFrame, source_column: str, period: int) -> pd.DataFrame:
    df['target'] = df[source_column].diff(period).shift(-period)
    df = df.iloc[:-period]
    return df


#%%
def create_target_pos_neg_classes(df: pd.DataFrame, source_column: str, period: int) -> pd.DataFrame:
    if period > 0:
        df['target'] = df[source_column].diff(period).shift(-period)
    else:
        df['target'] = df[source_column].diff(period)
    df['target'] = df['target'].map(lambda x: 0 if x <= 0.0 else 1)
    if period > 0:
        df = df.iloc[:-period]
    return df

def create_target_four_classes(df: pd.DataFrame, source_column: str, period: int) -> pd.DataFrame:
    def __get_class(x):
        treshold = 0.08
        if x <= -treshold:
            return 0
        elif x > -treshold and x <= 0:
            return 1
        elif x > 0 and x <= treshold:
            return 2
        else:
            return 3

    if period > 0:
        df['target'] = df[source_column].diff(period).shift(-period)
    else:
        df['target'] = df[source_column].diff(period)
    df['target'] = df['target'].map(__get_class)
    if period > 0:
        df = df.iloc[:-period]
    return df
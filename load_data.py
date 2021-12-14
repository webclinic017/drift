#%%
import pandas as pd
import os
import numpy as np
from pandas.core.frame import DataFrame
from utils.technical_indicators import ROC, RSI, STOK, STOD
from typing import Literal
from sklearn.preprocessing import OneHotEncoder

#%%

def get_all_assets(path: str) -> list[str]:
    return [f.split('.')[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and not f.startswith('.')]


def load_data(path: str,
            target_asset: str,
            target_asset_lags: list[int],
            load_other_assets: bool,
            other_asset_lags: list[int],
            log_returns: bool,
            add_date_features: bool,
            own_technical_features: Literal['none', 'level1', 'level2'],
            other_technical_features: Literal['none', 'level1', 'level2'],
            exogenous_features: Literal['none', 'level1'],
            index_column: Literal['date', 'int'],
            method: Literal['regression', 'classification'],
            narrow_format: bool = False,
        ) -> tuple[pd.DataFrame, pd.Series]:
    
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and not f.startswith('.')]
    files = [f for f in files if load_other_assets == True or (load_other_assets == False and f.startswith(target_asset))]
    def is_target_asset(target_asset: str, file: str): return file.split('.')[0].startswith(target_asset)
    dfs = [__load_df(
        path=os.path.join(path,f),
        prefix=f.split('.')[0],
        log_returns=log_returns,
        technical_features=own_technical_features if is_target_asset(target_asset, f) else other_technical_features,
        lags= target_asset_lags if is_target_asset(target_asset, f) else other_asset_lags,
        narrow_format=narrow_format,
    ) for f in files]
    if narrow_format:
        dfs = pd.concat(dfs, axis=0).fillna(0.)
    else:
        dfs = pd.concat(dfs, axis=1).fillna(0.)

    dfs.index = pd.DatetimeIndex(dfs.index)

    if add_date_features:
        dfs = pd.concat([dfs, pd.get_dummies(dfs.index.day, drop_first=True, prefix="day_month").set_index(dfs.index)], axis=1)
        dfs = pd.concat([dfs, pd.get_dummies(dfs.index.dayofweek, drop_first=True, prefix="day_week").set_index(dfs.index)] , axis=1)
        dfs = pd.concat([dfs, pd.get_dummies(dfs.index.month, drop_first=True, prefix="month").set_index(dfs.index)], axis = 1)

    if index_column == 'int':
        dfs.reset_index(drop=True, inplace=True)

    if narrow_format:
        dfs = dfs.drop(index=dfs.index[0], axis=0)

    ## Create target 
    target_col = 'target'
    returns_col = target_asset + '_returns'
    if method == 'regression':
        dfs = __create_target_cum_forward_returns(dfs, returns_col, 1)
    elif method == 'classification':
        dfs = __create_target_classes(dfs, returns_col, 1, 'two')
        
    X = dfs.drop(columns=[target_col])
    y = dfs[target_col]

    return X, y

def __load_df(path: str, prefix: str, log_returns: bool, technical_features: Literal['none', 'level1', 'level2'], lags: list[int], narrow_format: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, index_col=0).fillna(0)

    if log_returns:
        df['returns'] = np.log(df['close']).diff(1)
    else:
        df['returns'] = df['close'].pct_change()
    
    for lag in lags:
        df[f'lag_{lag}'] = df['returns'].shift(lag)

    df = __augment_derived_features(df, log_returns=log_returns, technical_features=technical_features)

    df = df.replace([np.inf, -np.inf], 0.)
    df = df.drop(columns=['open', 'high', 'low', 'close'])
    # we're not ready for this just yet
    if 'volume' in df.columns:
        df = df.drop(columns=['volume'])
    
    if narrow_format:
        df["ticker"] = np.repeat(prefix, df.shape[0])
    else: 
        df.columns = [prefix + "_" + c for c in df.columns]
    return df

def __augment_derived_features(df: pd.DataFrame, log_returns: bool, technical_features: Literal['none', 'level1', 'level2']) -> pd.DataFrame:
    if technical_features == 'level1' or technical_features == 'level2':
        # volatility (10, 20, 30 days)
        df['vol_10'] = df['returns'].rolling(10).std()*(252**0.5)
        df['vol_20'] = df['returns'].rolling(20).std()*(252**0.5)
        df['vol_30'] = df['returns'].rolling(30).std()*(252**0.5)
        df['vol_60'] = df['returns'].rolling(60).std()*(252**0.5)

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

    if technical_features == 'level2':
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
    return df


# %%

# %%
def __create_target_cum_forward_returns(df: pd.DataFrame, source_column: str, period: int) -> pd.DataFrame:
    df['target'] = df[source_column].diff(period).shift(-period)
    df = df.iloc[:-period]
    return df


def __create_target_classes(df: pd.DataFrame, source_column: str, period: int, no_of_classes: Literal["two", "three"]) -> pd.DataFrame:

    def get_class_binary(x):
        return 0 if x <= 0.0 else 1

    def get_class_threeway(x):
        bins = pd.qcut(df[source_column], 4, duplicates='raise', retbins=True)[1]
        lower_threshold = bins[1]
        upper_threshold = bins[3]
        if x <= lower_threshold:
            return -1
        elif x > lower_threshold and x < upper_threshold:
            return 0
        else:
            return 1

    if period > 0:
        df['target'] = df[source_column].shift(-period)
    else:
        df['target'] = df[source_column]
    
    get_class_function = get_class_binary
    if no_of_classes == "three":
        get_class_function = get_class_threeway

    df['target'] = df['target'].map(get_class_function)
    if period > 0:
        df = df.iloc[:-period]
    return df
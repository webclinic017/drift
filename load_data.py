#%%
import pandas as pd
import os
import numpy as np

#%%

def load_files(path: str, add_features: bool, log_returns: bool) -> pd.DataFrame:
    dfs = [__load_df(os.path.join(path,f), f.split('.')[0], add_features, log_returns) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    dfs = pd.concat(dfs, axis=1).fillna(0.)

    dfs.index = pd.DatetimeIndex(dfs.index)

    if add_features:
        dfs['day_month'] = dfs.index.day
        dfs['day_week'] = dfs.index.dayofweek
        dfs['month'] = dfs.index.month

    return dfs.drop(index=dfs.index[0], axis=0)

def __load_df(path: str, prefix: str, add_features: bool, log_returns: bool) -> pd.DataFrame:
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

    df = df.drop(columns=['open', 'high', 'low', 'close'])
    df.columns = [prefix + "_" + c for c in df.columns]
    return df

# %%

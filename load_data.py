#%%
import pandas as pd
import os

#%%

def load_files(path, add_features):
    dfs = [__load_df(os.path.join(path,f), f.split('.')[0], add_features) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    dfs = pd.concat(dfs, axis=1).fillna(0.)
    return dfs.drop(index=dfs.index[0], axis=0)

def __load_df(path, prefix, add_features):
    df = pd.read_csv(path, header=0, index_col=0).fillna(0)
    df['returns'] = df['close'].pct_change()
    
    if add_features:
        # volatility (10, 20, 30 days)
        df['vol_10'] = df['returns'].rolling(10).std()*(252**0.5)
        df['vol_20'] = df['returns'].rolling(20).std()*(252**0.5)
        df['vol_30'] = df['returns'].rolling(30).std()*(252**0.5)

        # momentum (10, 20, 30, 60, 90 days)
        df['mom_10'] = df['close'].pct_change(10)
        df['mom_20'] = df['close'].pct_change(20)
        df['mom_30'] = df['close'].pct_change(30)
        df['mom_60'] = df['close'].pct_change(60)
        df['mom_90'] = df['close'].pct_change(90)

    df = df.drop(columns=['open', 'high', 'low', 'close'])
    df.columns = [prefix + "_" + c for c in df.columns]
    return df

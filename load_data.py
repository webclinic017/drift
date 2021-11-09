#%%
import pandas as pd
import os


#%%

def load_files(path):
    dfs = [__load_df(os.path.join(path,f), f.split('.')[0]) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    return pd.concat(dfs, axis=1)

def __load_df(path, prefix):
    df = pd.read_csv(path, header=0, index_col=0).fillna(0)
    df = df.drop(columns=['open', 'high', 'low'])
    df.columns = [prefix + "_" + c for c in df.columns]
    return df

# %%
load_files("data/")

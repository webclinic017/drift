#%%
import pandas as pd
from tqdm import tqdm
from utils.glassnode import GlassnodeClient, Indicators, Mining

path = "data/daily_glassnode/"
client = GlassnodeClient(asset="BTC", since="2014-01-01", until="2021-12-17")
print("Client initiated")
indicator_client = Indicators(client)

indicator_names = [
    "rhodl_ratio",
    "cvdd",
    "difficulty_ribbon_compression",
    "nvt_ratio",
    "nvt_signal",
    "velocity",
    "supply_adjusted_cdd",
    "binary_cdd",
    "supply_adjusted_dormancy",
    "puell_multiple",
    "asopr",
    "reserve_risk",
    "sopr",
    "cdd",
    "asol",
    "msol",
    "dormancy",
    "liveliness",
    "relative_unrealized_profit",
    "relative_unrealized_loss",
    "nupl",
    # 'sth_nupl',
    # 'lth_nupl',
    "ssr",
    "bvin",
]


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    df.index.rename("time", inplace=True)
    if len(df.columns) != 1:
        df = df[["v"]]
    df.rename(columns={df.columns[0]: "close"}, inplace=True)
    df.sort_index(inplace=True)
    return df


for name in tqdm(indicator_names):
    method_to_call = getattr(indicator_client, name)
    df = method_to_call()
    df = process_df(df)
    df.to_csv(path + name + ".csv")


# Mining data

mining_names = ["hash_rate"]
mining_client = Mining(client)

for name in tqdm(mining_names):
    method_to_call = getattr(mining_client, name)
    df = method_to_call()
    df = process_df(df)
    df.to_csv(path + name + ".csv")

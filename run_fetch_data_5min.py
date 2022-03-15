from binance_historical_data import BinanceDataDumper
import datetime

data_dumper = BinanceDataDumper(
    path_dir_where_to_dump="./data/5min_crypto/",
    data_frequency="5m",
)

assets = [
    "TRXUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "SOLUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "ETHUSDT",
    "LTCUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "ETCUSDT",
]

data_dumper.dump_data(
    list_tickers=assets,
    date_start=datetime.date(2018, 1, 1),
    date_end=datetime.date(2022, 1, 1),
    is_to_update_existing=False,
)

import os
from tqdm import tqdm
import pandas as pd

for asset in tqdm(assets):
    path = f"./data/5min_crypto/spot/monthly/klines/{asset}/5m/"
    files = os.listdir(path)

    def load_df(path):
        df = pd.read_csv(
            path,
            names=[
                "timestamp",
                "open",
                "low",
                "high",
                "close",
                "volume",
                "Closetime",
                "Quote asset volume",
                "Number of trades",
                "Taker buy base asset volume",
                "Taker buy quote asset volume",
                "Ignore",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df

    dfs = pd.concat([load_df(path + file) for file in files], axis=0)
    dfs.sort_index(inplace=True)
    dfs.to_parquet(f"./data/5min_crypto/{asset}.parquet")

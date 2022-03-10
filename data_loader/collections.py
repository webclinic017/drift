from typing import Literal
from .types import DataSource, DataCollection


def transform_to_data_collection(
    path: str, file_names: list[str], freq: Literal["5m", "1h", "1d"]
) -> DataCollection:
    return [DataSource(path, file_name, freq) for file_name in file_names]


__daily_etf = ["GLD", "IEF", "QQQ", "SPY", "TLT"]

__5min_crypto = [
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

__daily_glassnode = [
    "rhodl_ratio",
    # 'cvdd',
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
    "sth_nupl",
    "lth_nupl",
    "ssr",
    "bvin",
    # 'hash_rate'
]


data_collections = dict(
    daily_etf=transform_to_data_collection("data/daily_etf", __daily_etf, "1d"),
    fivemin_crypto=transform_to_data_collection(
        "data/5min_crypto", __5min_crypto, "5m"
    ),
    daily_glassnode=transform_to_data_collection(
        "data/daily_glassnode", __daily_glassnode, "1d"
    ),
)

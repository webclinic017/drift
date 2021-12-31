from utils.types import Path, FileName, DataSource, DataCollection
from utils.helpers import flatten

daily_etf = ["GLD", "IEF", "QQQ", "SPY", "TLT"]
daily_etf = list(zip(["data/daily_etf"] * len(daily_etf), daily_etf))

daily_crypto = ["ADA_USD",
    "BCH_USD",
    "BNB_USD",
    "BTC_USD",
    "DOT_USD",
    "ETC_USD",
    "ETH_USD",
    "FIL_USD",
    "LTC_USD",
    "SOL_USD",
    "THETA_USD",
    "TRX_USD",
    "UNI_USD",
    "XLM_USD",
    "XRP_USD",
    "XTZ_USD"]
daily_crypto = list(zip(["data/daily_crypto"] * len(daily_crypto), daily_crypto))


hourly_crypto = ["BTC_USD", "DASH_USD", "ETC_USD", "ETH_USD", "LTC_USD", "TRX_USD", "XLM_USD", "XMR_USD", "XRP_USD"]
hourly_crypto = list(zip(["data/hourly_crypto"] * len(hourly_crypto), hourly_crypto))


data_collections = dict(
    daily_crypto = daily_crypto,
    daily_etf = daily_etf,
    hourly_crypto = hourly_crypto
)

def preprocess_data_collections_config(data_dict: dict) -> dict:
    data_dict = data_dict.copy()
    keys = ['assets', 'other_assets']
    # keys = ['assets', 'other_assets', 'exogenous_data']
    for key in keys:
        preset_names = data_dict[key]
        data_dict[key] = flatten([data_collections[preset_name] for preset_name in preset_names])
    return data_dict

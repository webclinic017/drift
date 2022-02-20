#%%
import pandas as pd
from data_loader.get_prices import get_crypto_price_crypto_compare, get_stock_price_av

#%%

etf_tickers = ["GLD", "IEF", "TLT", "SPY", "QQQ"]
etf_path = "data/daily_etf"

#%%
for ticker in etf_tickers:
    print("Fetching ", ticker)
    df = get_stock_price_av(ticker, "2017-11-10")

    df.to_csv(f"{etf_path}/{ticker}.csv", index=True)

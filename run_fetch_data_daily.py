#%%
import pandas as pd
from data_loader.get_prices import get_crypto_price_crypto_compare, get_stock_price_av

#%%
crypto_tickers = ["BTC", "ETH", "BNB", "ADA", "SOL", "XRP", "DOT", "LTC", "UNI", "TRX", "XLM", "BCH", "FIL", "ETC", "THETA", "XTZ"]
etf_tickers = ["GLD", "IEF", "TLT", "SPY", "QQQ"]
crypto_path = "data/daily_crypto"
etf_path = "data/daily_etf"

#%%
for ticker in etf_tickers:
    print("Fetching ", ticker)
    df = get_stock_price_av(ticker, "2017-11-10")

    df.to_csv(f"{etf_path}/{ticker}.csv", index=True)

for src_ticker in crypto_tickers:
    print("Fetching ", src_ticker, "USD")
    df = get_crypto_price_crypto_compare(src_ticker, "USD", 1500)

    df.to_csv(f"{crypto_path}/{src_ticker}_USD.csv", index=True)



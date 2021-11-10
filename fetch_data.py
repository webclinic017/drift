#%%
import pandas as pd
from utils.get_prices import get_crypto_price_crypto_compare, get_stock_price_av

#%%
crypto_tickers = ["BTC", "ETH", "BNB", "ADA", "SOL", "XRP", "DOT", "LTC", "UNI", "TRX", "FIL"]
etf_tickers = ["GLD", "IEF", "TLT", "SPY", "QQQ"]
tickers = crypto_tickers + etf_tickers

#%%
for ticker in tickers:
    print("Fetching ", ticker)
    if ticker in crypto_tickers:
        df = get_crypto_price_crypto_compare(ticker, "USD", 1500)
    else:
        df = get_stock_price_av(ticker, "2017-11-10")

    df.to_csv(f"data/{ticker}.csv", index=True)


# %%

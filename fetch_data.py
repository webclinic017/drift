#%%
import pandas as pd
from utils.get_prices import get_crypto_price_crypto_compare, get_stock_price_av
from itertools import combinations

#%%
crypto_tickers = ["BTC", "ETH", "BNB", "ADA", "SOL", "XRP", "DOT", "LTC", "UNI", "TRX", "FIL", "USD"]
etf_tickers = ["GLD", "IEF", "TLT", "SPY", "QQQ"]
tickers = crypto_tickers + etf_tickers

#%%
for ticker in etf_tickers:
    print("Fetching ", ticker)
    df = get_stock_price_av(ticker, "2017-11-10")

    df.to_csv(f"data/{ticker}.csv", index=True)

crypto_ticker_pairs = list(combinations(crypto_tickers, 2))
for src_ticker, trg_ticker in crypto_ticker_pairs:
    print("Fetching ", src_ticker, trg_ticker)
    df = get_crypto_price_crypto_compare(src_ticker, trg_ticker, 1500)

    df.to_csv(f"data/{src_ticker}_{trg_ticker}.csv", index=True)


# %%

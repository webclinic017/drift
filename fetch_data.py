#%%
import pandas as pd
from utils.get_prices import get_crypto_price_crypto_compare, get_stock_price_av

#%%
crypto_tickers = ["BTC", "ETH", "BNB", "ADA", "SOL", "XRP", "DOT", "LTC", "UNI", "TRX", "FIL"]
etf_tickers = ["GLD", "IEF", "TLT", "SPY", "QQQ"]
tickers = crypto_tickers + etf_tickers

add_features = True

#%%
for ticker in tickers:
    print("Fetching ", ticker)
    if ticker in crypto_tickers:
        df = get_crypto_price_crypto_compare(ticker, "USD", 1500)
    else:
        df = get_stock_price_av(ticker, "2017-11-10")
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

    df.to_csv(f"data/{ticker}.csv", index=True)


# %%

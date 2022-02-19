# %%
import requests
import pandas as pd

# %%
def get_crypto_price_crypto_compare(
    symbol: str, exchange: str, days: int
) -> pd.DataFrame:
    api_url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym={exchange}&limit={days}&api_key={CC_API_KEY}"
    raw = requests.get(api_url).json()
    df = pd.DataFrame(raw["Data"]["Data"])[
        ["time", "high", "low", "open", "close"]
    ].set_index("time")
    df.index = pd.to_datetime(df.index, unit="s")
    df.sort_index(inplace=True, ascending=True)
    return df


def get_crypto_price_av(symbol: str, exchange: str, start_date=None) -> pd.DataFrame:
    api_url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market={exchange}&apikey={AV_API_KEY}"
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df["Time Series (Digital Currency Daily)"]).T
    df = df.rename(
        columns={
            "1a. open (USD)": "open",
            "2a. high (USD)": "high",
            "3a. low (USD)": "low",
            "4a. close (USD)": "close",
            "5. volume": "volume",
        }
    )
    for i in df.columns:
        df[i] = df[i].astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.iloc[::-1].drop(
        [
            "1b. open (USD)",
            "2b. high (USD)",
            "3b. low (USD)",
            "4b. close (USD)",
            "6. market cap (USD)",
        ],
        axis=1,
    )
    if start_date:
        df = df[df.index >= start_date]
    df.sort_index(inplace=True, ascending=True)
    return df


def get_stock_price_av(symbol: str, start_date: str = None) -> pd.DataFrame:
    api_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={AV_API_KEY}"
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df["Time Series (Daily)"]).T
    df = df.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "5. adjusted close": "close",
            "6. volume": "volume",
        }
    )
    for i in df.columns:
        df[i] = df[i].astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.iloc[::-1].drop(
        ["4. close", "7. dividend amount", "8. split coefficient"], axis=1
    )
    if start_date:
        df = df[df.index >= start_date]
    df.sort_index(inplace=True, ascending=True)
    df = df.rename_axis("time")
    return df

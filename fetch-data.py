# %%
import requests
import pandas as pd

AV_API_KEY = 'UY5VGSWBE88SHGI6'
CC_API_KEY = 'bfb8b5f54b21354608020a6654b370617b2fcabd2c8c2ce04ab881682a1d9dc9'

# %%
def get_crypto_price_crypto_compare(symbol, exchange, days):
    api_url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym={exchange}&limit={days}&api_key={CC_API_KEY}'
    raw = requests.get(api_url).json()
    df = pd.DataFrame(raw['Data']['Data'])[['time', 'high', 'low', 'open']].set_index('time')
    df.index = pd.to_datetime(df.index, unit = 's')
    return df

ada = get_crypto_price_crypto_compare('ADA', 'USD', 1500)
ada


# %%

def get_crypto_price_av(symbol, exchange, start_date = None):
    api_url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market={exchange}&apikey={AV_API_KEY}'
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df['Time Series (Digital Currency Daily)']).T
    df = df.rename(columns = {'1a. open (USD)': 'open', '2a. high (USD)': 'high', '3a. low (USD)': 'low', '4a. close (USD)': 'close', '5. volume': 'volume'})
    for i in df.columns:
        df[i] = df[i].astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.iloc[::-1].drop(['1b. open (USD)', '2b. high (USD)', '3b. low (USD)', '4b. close (USD)', '6. market cap (USD)'], axis = 1)
    if start_date:
        df = df[df.index >= start_date]
    return df

btc = get_crypto_price_av(symbol = 'BTC', exchange = 'USD', start_date = '2018-01-01')
btc



# %%

# %%

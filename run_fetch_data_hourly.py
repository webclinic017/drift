import pandas as pd
import ssl
from tqdm import tqdm

base_url = "https://www.cryptodatadownload.com/cdd/"

exchange_name = "Bitfinex"
files_to_download = [
 exchange_name + '_TRXUSD_1h.csv',
 exchange_name + '_ETHUSD_1h.csv',
 exchange_name + '_XLMUSD_1h.csv',
 exchange_name + '_XMRUSD_1h.csv',
 exchange_name + '_LTCUSD_1h.csv',
#  exchange_name + '_FILUSD_1h.csv',
 exchange_name + '_DASHUSD_1h.csv',
#  exchange_name + '_LINKUSD_1h.csv',
#  exchange_name + '_SOLUSD_1h.csv',
 exchange_name + '_BTCUSD_1h.csv',
 exchange_name + '_ETCUSD_1h.csv',
#  exchange_name + '_VETUSD_1h.csv',
#  exchange_name + '_DOTUSD_1h.csv'
 ]


for file in tqdm(files_to_download):
    data_location = base_url + file
    ssl._create_default_https_context = ssl._create_unverified_context
    data = pd.read_csv(data_location, skiprows=1, index_col=1, parse_dates=True).drop(columns=['unix'])
    volume_column_to_delete = [c for c in data.columns if c.startswith('Volume') and 'USD' not in c]
    data.drop(volume_column_to_delete + ['symbol'], axis=1, inplace=True)
    data.rename({'Volume USD': 'volume'}, axis=1, inplace=True)
    data.index.rename('time', inplace=True)
    target_file = file.split('_')[1].replace('USD', '') + '_USD'
    data.to_csv(f"data/hourly_crypto/{target_file}.csv")

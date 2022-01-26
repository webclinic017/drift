import pandas as pd
import ssl
from tqdm import tqdm
from data_loader.utils import deduplicate_indexes

base_url = "https://www.cryptodatadownload.com/cdd/"

exchange_name = "Bitfinex"
period = "minute" # 1h
files_to_download = [
 exchange_name + '_TRXUSD_' + period + '.csv',
 exchange_name + '_ETHUSD_' + period + '.csv',
 exchange_name + '_XLMUSD_' + period + '.csv',
 exchange_name + '_XMRUSD_' + period + '.csv',
 exchange_name + '_LTCUSD_' + period + '.csv',
 exchange_name + '_DASHUSD_' + period + '.csv',
 exchange_name + '_BTCUSD_' + period + '.csv',
 exchange_name + '_ETCUSD_' + period + '.csv',
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
    data = deduplicate_indexes(data)
    data.to_csv(f"data/minute_crypto/{target_file}.csv")

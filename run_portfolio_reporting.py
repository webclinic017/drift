#%%
import pandas as pd
import numpy as np
from data_loader.load_data import load_only_returns
from data_loader.collections import data_collections
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, CallSeqType
from utils.helpers import get_first_valid_return_index
from alphalens.tears import (create_returns_tear_sheet,
                      create_information_tear_sheet,
                      create_turnover_tear_sheet,
                      create_summary_tear_sheet,
                      create_full_tear_sheet,
                      create_event_returns_tear_sheet,
                      create_event_study_tear_sheet)

from alphalens.utils import get_clean_factor_and_forward_returns

def fixed_weight(row: pd.Series, availability_row: pd.Series, allow_short: bool) -> pd.Series:
    no_of_assets_available = availability_row.sum()
    unit = 1 / no_of_assets_available
    if allow_short:
        def determine_pos(x):
            if x == 0.0 or np.isnan(x):
                return 0
            elif x > 0.0:
                return 1
            else:
                return -1
        row = row.apply(determine_pos)
    else:
        row = row.apply(lambda x: 1 if x > 0.0 else 0)
    return row * unit

def limit_weight(row: pd.Series) -> pd.Series:
    if row.sum() > 1:
        row = row / row.sum()
    elif row.sum() < 1:
        row = row * (1. / abs(row.sum()))
    return row

def only_top_bottom_2(row: pd.Series) -> pd.Series:
    row = row.copy()
    row_sorted = row.sort_values()
    bottom = row_sorted.iloc[:2]
    top = row_sorted.iloc[-2:]
    middle = row_sorted.iloc[2:-2]
    for index, _ in middle.iteritems():
        row[index] = 0.

    if bottom.sum() > 0:
        for index, _ in bottom.iteritems():
            row[index] = 0.
    else:
        for index, _ in bottom.iteritems():
            row[index] = -.025

    if top.sum() < 0:
        for index, _ in top.iteritems():
            row[index] = 0.
    else:
        for index, _ in top.iteritems():
            row[index] = .025
    
    return row
    
def equal_weight(row: pd.Series, availability: pd.Series) -> pd.Series:
    no_of_assets_available = availability.sum()
    unit = 1 / no_of_assets_available
    for index, _ in predictions.iteritems():
        row[index] = 0. if availability[index] == 0 else unit
    return row



def create_naive_portfolio_weights(predictions: pd.DataFrame, availability: pd.DataFrame, allow_short: bool) -> pd.DataFrame:
    weights = predictions.copy()
    assert weights.shape[1] == availability.shape[1]
    for index, row in weights.iterrows():
        # row = fixed_weight(row, availability.iloc[index], allow_short)
        # row = row / row.sum()
        # row = only_top_bottom_2(row)
        # row = equal_weight(row, availability.iloc[index])
        row = limit_weight(row)
        weights.iloc[index] = row
    return weights


predictions = pd.read_csv('output/predictions.csv', index_col=0)
predictions.columns = ['_'.join(col.replace("model_", "").split("_")[:2]) for col in predictions.columns]
first_index = get_first_valid_return_index(predictions[predictions.columns[0]])
predictions = predictions.iloc[first_index:]
predictions.reset_index(drop=True, inplace=True)

close = load_only_returns(data_collections['daily_crypto'], 'date', 'price')
close = close.iloc[first_index:-1]
close.columns = [col.replace("_returns", "") for col in close.columns]
close = close[predictions.columns]
close.reset_index(drop=True, inplace=True)

# returns = load_only_returns(data_collections['daily_crypto'], 'date', 'returns')
# returns = returns.iloc[first_index:-1]
# returns.columns = [col.replace("_returns", "") for col in returns.columns]
# returns = returns[predictions.columns]
# returns.reset_index(drop=True, inplace=True)

# predictions = predictions.reindex(close.index, method='ffill')
availability = close.applymap(lambda x: 0 if x == 0.0 or x == 0 or np.isnan(x) else 1)

weights = create_naive_portfolio_weights(predictions, availability, allow_short=True)
# weights_long = pd.melt(weights,id_vars=['index'])



# factor_data = get_clean_factor_and_forward_returns(
#     weights,
#     close,
#     groupby=factor_groups,
#     quantiles=4,
#     periods=(1, 3), 
#     filter_zscore=None)


# rebalance every n days
# weights.iloc[np.arange(len(weights)) % 7 != 0] = np.nan



# portfolio = vbt.Portfolio.from_orders(
#     close=close,
#     size=weights,
#     size_type=SizeType.TargetPercent,
#     cash_sharing=True,
#     call_seq=CallSeqType.Auto,
#     group_by=True,
#     freq='1D',
#     raise_reject=True,
#     fees=0.01,
#     seed=1,
#     init_cash=1e5,
# )

# print(portfolio.stats())

# from pypfopt import risk_models
# from pypfopt import expected_returns
# from pypfopt import EfficientFrontier
# mu = expected_returns.mean_historical_return(close)
# S = risk_models.sample_cov(close)

# ef = EfficientFrontier(mu, S)
# raw_weights = ef.max_sharpe()
# cleaned_weights = ef.clean_weights()
# print(ef.portfolio_performance(verbose=True))

# %%

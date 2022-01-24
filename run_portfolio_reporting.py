#%%
import pandas as pd
import numpy as np
from data_loader.load_data import load_only_returns
from data_loader.collections import data_collections

from utils.helpers import get_first_valid_return_index
import alphalens
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, CallSeqType, Direction
import quantstats as qs

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
    elif row.sum() < -1:
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

def create_quantile_weights(predictions: pd.DataFrame, availability: pd.DataFrame, allow_short: bool) -> pd.DataFrame:
    weights = predictions.copy()
    assert weights.shape[1] == availability.shape[1]
    quantiles = weights.copy()
    for column in weights.columns:
        quantiles[column] = pd.qcut(weights[column], q=4, labels=False, duplicates='drop')
    
    for index, row in quantiles.iterrows():
        def only_select_bottom_top(x):
            if x == 0:
                return -1
            elif x == 3:
                return 1
            else:
                return 0
        row = row.apply(only_select_bottom_top)
        no_of_nonzero_predictions = row[row != 0].count()
        units = min(1 / no_of_nonzero_predictions, 0.25)
        weights.loc[index] = row * units
    return weights


predictions = pd.read_csv('output/predictions.csv', index_col=0)
predictions.index = pd.DatetimeIndex(predictions.index)
predictions.columns = ['_'.join(col.replace("model_", "").split("_")[:2]) for col in predictions.columns]
first_index = get_first_valid_return_index(predictions[predictions.columns[0]])
predictions = predictions.iloc[first_index:]

close = load_only_returns(data_collections['daily_crypto'], 'price')
close = close.iloc[first_index:-1]
close.columns = [col.replace("_returns", "") for col in close.columns]
close = close[predictions.columns]

availability = close.applymap(lambda x: 0 if x == 0.0 or x == 0 or np.isnan(x) else 1)

def report_alphalens():
    alpha_factors = predictions.copy()
    alpha_factors.index = close.index

    alpha_factors_long = pd.melt(alpha_factors.reset_index(), id_vars=['time'], value_vars=alpha_factors.columns).set_index(['time', 'variable'])
    factor_data = get_clean_factor_and_forward_returns(
        alpha_factors_long,
        close,
        quantiles=4,
        periods=(1, 2, 3, 4, 5, 6, 10), 
        filter_zscore=None)
    # create_full_tear_sheet(factor_data, long_short=True)

    from matplotlib.backends.backend_pdf import PdfPages

    mean_return_by_q_daily, std_err = alphalens.performance.mean_return_by_quantile(factor_data, by_date=True)
    mean_return_by_q, std_err_by_q = alphalens.performance.mean_return_by_quantile(factor_data, by_group=False)
    plot1 = alphalens.plotting.plot_quantile_returns_bar(mean_return_by_q)
    plot2 = alphalens.plotting.plot_quantile_returns_violin(mean_return_by_q_daily)
    plot3 = alphalens.plotting.plot_cumulative_returns_by_quantile(mean_return_by_q_daily, period='D')


    with PdfPages('output/factors.pdf') as pdf:
        pdf.savefig(plot1.figure)
        pdf.savefig(plot2.figure)
        pdf.savefig(plot3.figure)



def report_backtest() -> vbt.Portfolio:
    weights = create_quantile_weights(predictions, availability, allow_short=True)

    weights.index = close.index
    # rebalance every n days
    # weights.iloc[np.arange(len(weights)) % 2 != 0] = np.nan

    weights.to_csv('output/weights.csv')

    portfolio = vbt.Portfolio.from_orders(
        close=close,
        size=weights,
        size_type=SizeType.TargetPercent,
        direction=Direction.Both,
        cash_sharing=True,
        call_seq=CallSeqType.Auto,
        group_by=True,
        freq='1D',
        raise_reject=True,
        fees=0.001, # assuming 0.1% fees (1.5x of FTX)
        slippage= 0.002, # assuming 0.2% slippage (5x of avg. spread on FTX)
        seed=1,
        init_cash=1e5,
        log=True
    )

    qs.reports.full(portfolio.returns(), portfolio.benchmark_returns()) 

    qs.reports.html(portfolio.returns(), portfolio.benchmark_returns(), output='output/report.html') 
    print(portfolio.stats())
    return portfolio

portfolio = report_backtest()




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

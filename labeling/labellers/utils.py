import pandas as pd
from data_loader.types import ForwardReturnSeries


def create_forward_returns(series: pd.Series, period: int) -> ForwardReturnSeries:
    assert period > 0
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=period)

    forward_returns = series.rolling(window=indexer).sum()
    return forward_returns

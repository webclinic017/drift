import pandas as pd
from data_loader.types import ForwardReturnSeries
from labeling.types import EventsDataFrame
from typing import Callable
import numpy as np


def create_forward_returns(series: pd.Series, period: int) -> ForwardReturnSeries:
    assert period > 0
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=period)

    forward_returns = series.rolling(window=indexer).sum()
    return forward_returns


def purge_overlapping_events(events: EventsDataFrame) -> EventsDataFrame:
    events = events.copy()
    indicies_to_remove = []
    last_event_end = events.iloc[0]["start"]
    for index, row in events.iterrows():
        if row["start"] < last_event_end:
            indicies_to_remove.append(index)
        else:
            last_event_end = row["end"]
    events.drop(indicies_to_remove, inplace=True)
    return events


def discretize_binary(x):
    return 1 if x > 0 else -1


def discretize_binary_zero_one(x):
    return 1 if x > 0 else 0


def discretize_threeway(x):
    return 0 if x == 0 else 1 if x > 0 else -1


def discretize_threeway_threshold(threshold: float) -> Callable:
    def discretize(current_value):
        lower_threshold = -threshold
        upper_threshold = threshold
        if np.isnan(current_value):
            return np.nan
        elif current_value <= lower_threshold:
            return -1
        elif current_value > lower_threshold and current_value < upper_threshold:
            return 0
        else:
            return 1

    return discretize

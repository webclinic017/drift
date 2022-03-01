import pandas as pd
from data_loader.types import ForwardReturnSeries
from labeling.types import EventsDataFrame


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

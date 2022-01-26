

from ..types import EventFilter
from data_loader.types import ReturnSeries
import pandas as pd

class CUSUMVolatilityEventFilter(EventFilter):

    def __init__(self, vol_period: int):
        self.vol_period = vol_period

    def get_event_start_times(self, returns: ReturnSeries) -> pd.DatetimeIndex:

        rolling_vol = returns.rolling(self.vol_period).std() * 0.15

        filtered_indices = []
        pos_threshold = 0
        neg_threshold = 0
        diff = returns.diff()
        for index in diff.index[1:]:
            pos_threshold, neg_threshold = (
                max(0, pos_threshold + diff.loc[index]),
                min(0, neg_threshold + diff.loc[index]),
            )

            if neg_threshold < -rolling_vol[index]:
                neg_threshold = 0
                filtered_indices.append(index)

            elif pos_threshold > rolling_vol[index]:
                pos_threshold = 0
                filtered_indices.append(index)

        return pd.DatetimeIndex(filtered_indices)

class CUSUMFixedEventFilter(EventFilter):

    def __init__(self, threshold: float):
        self.threshold = threshold

    def get_event_start_times(self, returns: ReturnSeries) -> pd.DatetimeIndex:
        filtered_indices = []
        pos_threshold = 0
        neg_threshold = 0
        diff = returns.diff()
        for index in diff.index[1:]:
            pos_threshold, neg_threshold = (
                max(0, pos_threshold + diff.loc[index]),
                min(0, neg_threshold + diff.loc[index]),
            )

            if neg_threshold < -self.threshold:
                neg_threshold = 0
                filtered_indices.append(index)

            elif pos_threshold > self.threshold:
                pos_threshold = 0
                filtered_indices.append(index)

        return pd.DatetimeIndex(filtered_indices)
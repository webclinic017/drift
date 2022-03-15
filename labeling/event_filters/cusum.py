from numpy import float32
from ..types import EventFilter
from data_loader.types import ReturnSeries
import pandas as pd
from numba import njit
from numba.typed import List


class CUSUMVolatilityEventFilter(EventFilter):
    def __init__(self, vol_period: int, multiplier: float):
        self.vol_period = vol_period
        self.multiplier = multiplier

    def get_event_start_times(self, returns: ReturnSeries) -> pd.DatetimeIndex:
        rolling_vol = returns.rolling(self.vol_period).std() * self.multiplier
        diffed_returns = returns.diff()

        int_indicies = _process_vol_based(
            List(diffed_returns.to_list()), List(rolling_vol.to_list())
        )

        return pd.DatetimeIndex([returns.index[i] for i in int_indicies])


class CUSUMFixedEventFilter(EventFilter):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def get_event_start_times(self, returns: ReturnSeries) -> pd.DatetimeIndex:
        diffed_returns = returns.diff()
        int_indicies = _process_fixed(
            List(diffed_returns.to_list()), abs(returns.mean()) * self.threshold
        )

        return pd.DatetimeIndex([returns.index[i] for i in int_indicies])


@njit
def _process_fixed(diffed_returns: List, threshold: float32) -> List:
    pos_threshold: float32 = 0.0  # type: ignore
    neg_threshold: float32 = 0.0  # type: ignore
    filtered_indicies = List()
    for index in range(1, len(diffed_returns[1:])):
        pos_threshold, neg_threshold = (  # type: ignore
            max(0, pos_threshold + diffed_returns[index]),  # type: ignore
            min(0, neg_threshold + diffed_returns[index]),  # type: ignore
        )

        if neg_threshold < -threshold:
            neg_threshold = 0.0  # type: ignore
            filtered_indicies.append(index)

        elif pos_threshold > threshold:
            pos_threshold = 0.0  # type: ignore
            filtered_indicies.append(index)

    return filtered_indicies


@njit
def _process_vol_based(diffed_returns: List, rolling_vol: List) -> List:
    pos_threshold: float32 = 0.0  # type: ignore
    neg_threshold: float32 = 0.0  # type: ignore
    filtered_indicies = List()
    for index in range(1, len(diffed_returns[1:])):
        pos_threshold, neg_threshold = (  # type: ignore
            max(0, pos_threshold + diffed_returns[index]),  # type: ignore
            min(0, neg_threshold + diffed_returns[index]),  # type: ignore
        )

        if neg_threshold < -rolling_vol[index]:
            neg_threshold = 0.0  # type: ignore
            filtered_indicies.append(index)

        elif pos_threshold > rolling_vol[index]:
            pos_threshold = 0.0  # type: ignore
            filtered_indicies.append(index)

    return filtered_indicies

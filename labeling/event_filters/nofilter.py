from ..types import EventFilter
from data_loader.types import ReturnSeries
import pandas as pd


class NoEventFilter(EventFilter):
    def __init__(self, ignored_threshold_multiplier: float):
        pass

    def get_event_start_times(self, returns: ReturnSeries) -> pd.DatetimeIndex:
        return returns.index

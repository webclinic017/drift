from data_loader.types import ReturnSeries, ForwardReturnSeries
from abc import ABC, abstractmethod
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series


class EventFilter(ABC):
    @abstractmethod
    def get_event_start_times(self, returns: ReturnSeries) -> pd.DatetimeIndex:
        raise NotImplementedError


class EventSchema(pa.SchemaModel):
    start: Series[pd.Timestamp]
    end: Series[pd.Timestamp]
    label: Series[int]
    returns: Series[float]


EventsDataFrame = DataFrame[EventSchema]


class EventLabeller(ABC):
    @abstractmethod
    def label_events(
        self, event_start_times: pd.DatetimeIndex, returns: ReturnSeries
    ) -> EventsDataFrame:
        raise NotImplementedError

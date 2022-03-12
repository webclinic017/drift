from ..types import EventLabeller, EventsDataFrame, ReturnSeries
import pandas as pd
from .utils import create_forward_returns
from typing import Callable
from .utils import discretize_binary


class FixedTimeHorionTwoClassEventLabeller(EventLabeller):

    time_horizon: int

    def __init__(self, time_horizon: int):
        self.time_horizon = time_horizon

    def label_events(
        self, event_start_times: pd.DatetimeIndex, returns: ReturnSeries
    ) -> EventsDataFrame:

        forward_returns = create_forward_returns(returns, self.time_horizon)
        cutoff_point = returns.index[-self.time_horizon]
        event_start_times = event_start_times[event_start_times < cutoff_point]
        event_candidates = forward_returns[event_start_times]

        def get_class_binary(x: float) -> int:
            return -1 if x <= 0.0 else 1

        labels = event_candidates.map(get_class_binary)
        events = pd.DataFrame(
            {
                "start": event_start_times,
                "end": event_start_times + pd.Timedelta(minutes=self.time_horizon * 5),
                "label": labels,
                "returns": forward_returns[event_start_times],
            }
        )
        return events

    def get_labels(self) -> list[int]:
        return [-1, 1]

    def get_discretize_function(self) -> Callable:
        return discretize_binary

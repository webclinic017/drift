from ..types import EventLabeller, EventsDataFrame, ReturnSeries, ForwardReturnSeries
import pandas as pd
from .utils import create_forward_returns


class FixedTimeHorionTwoClassEventLabeller(EventLabeller):

    time_horizon: int

    def __init__(self, time_horizon: int):
        self.time_horizon = time_horizon

    def label_events(
        self, event_start_times: pd.DatetimeIndex, returns: ReturnSeries
    ) -> tuple[EventsDataFrame, ForwardReturnSeries]:

        forward_returns = create_forward_returns(returns, self.time_horizon)
        cutoff_point = returns.index[-self.time_horizon]
        event_start_times[event_start_times < cutoff_point]
        event_candidates = forward_returns[event_start_times]

        def get_class_binary(x: float) -> int:
            return -1 if x <= 0.0 else 1

        labels = event_candidates.map(get_class_binary)

        return (
            pd.DataFrame(
                {
                    "start": event_start_times,
                    "end": event_start_times + pd.Timedelta(days=self.time_horizon),
                    "label": labels,
                    "returns": forward_returns[event_start_times],
                }
            ),
            forward_returns[event_start_times],
        )

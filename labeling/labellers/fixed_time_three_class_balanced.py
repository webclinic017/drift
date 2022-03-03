from data_loader.types import ReturnSeries
from ..types import EventLabeller, EventsDataFrame
import pandas as pd
from .utils import create_forward_returns
from typing import Callable


class FixedTimeHorionThreeClassBalancedEventLabeller(EventLabeller):

    time_horizon: int

    def __init__(self, time_horizon: int):
        self.time_horizon = time_horizon

    def label_events(
        self, event_start_times: pd.DatetimeIndex, returns: ReturnSeries
    ) -> EventsDataFrame:

        forward_returns = create_forward_returns(returns, self.time_horizon)
        cutoff_point = returns.index[-self.time_horizon]
        event_start_times[event_start_times < cutoff_point]
        event_candidates = forward_returns[event_start_times]

        def get_bins_threeway(x):
            bins = pd.qcut(event_candidates, 3, retbins=True, duplicates="drop")[1]

            if len(bins) != 4:
                # if we don't have enough data for the quantiles, we'll need to add hard-coded values
                lower_bound = bins[0]
                upper_bound = bins[-1]
                bins = [lower_bound] + [-0.02, 0.02] + [upper_bound]
            return bins

        bins = get_bins_threeway(event_candidates)

        def map_class_threeway(current_value):
            lower_threshold = bins[1]
            upper_threshold = bins[2]
            if current_value <= lower_threshold:
                return -1
            elif current_value > lower_threshold and current_value < upper_threshold:
                return 0
            else:
                return 1

        labels = event_candidates.map(map_class_threeway)
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
        return [-1, 0, 1]

    def get_discretize_function(self) -> Callable:
        raise NotImplementedError

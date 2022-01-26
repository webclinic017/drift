from data_loader.types import ForwardReturnSeries
from ..types import EventLabeller, EventsDataFrame
import pandas as pd

class FixedTimeHorionThreeClassBalancedEventLabeller(EventLabeller):

    time_horizon: int

    def __init__(self, time_horizon: int = 1):
        self.time_horizon = time_horizon

    def label_events(self, event_start_times: pd.DatetimeIndex, forward_returns: ForwardReturnSeries) -> EventsDataFrame:

        event_candidates = forward_returns[event_start_times]

        def get_bins_threeway(x):
            bins = pd.qcut(event_candidates, 3, retbins=True, duplicates = 'drop')[1]

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
        
        return pd.DataFrame({
            'start': event_start_times,
            'end': event_start_times + pd.Timedelta(days=self.time_horizon),
            'label': labels,
            'returns': forward_returns[event_start_times]
        })


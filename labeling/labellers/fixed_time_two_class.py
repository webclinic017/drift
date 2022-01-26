

from ..types import EventLabeller, EventsDataFrame, ForwardReturnSeries
import pandas as pd

class FixedTimeHorionTwoClassEventLabeller(EventLabeller):

    time_horizon: int

    def __init__(self, time_horizon: int = 1):
        self.time_horizon = time_horizon

    def label_events(self, event_start_times: pd.DatetimeIndex, forward_returns: ForwardReturnSeries) -> EventsDataFrame:

        event_candidates = forward_returns[event_start_times]

        def get_class_binary(x: float) -> int:
            return -1 if x <= 0.0 else 1
        labels = event_candidates.map(get_class_binary)
        
        return pd.DataFrame({
            'start': event_start_times,
            'end': event_start_times + pd.Timedelta(days=self.time_horizon),
            'label': labels,
            'returns': forward_returns[event_start_times]
        })




from .types import EventFilter, EventLabeller, EventsDataFrame
from data_loader.types import ForwardReturnSeries, XDataFrame, ReturnSeries, ySeries


def label_data(
    event_filter: EventFilter,
    event_labeller: EventLabeller,
    X: XDataFrame,
    returns: ReturnSeries,
) -> tuple[EventsDataFrame, XDataFrame, ySeries, ForwardReturnSeries]:

    event_start_times = event_filter.get_event_start_times(returns)
    print(
        "| Filtered out ",
        (1 - (len(event_start_times) / len(returns))) * 100,
        "% of timestamps",
    )

    events, forward_returns = event_labeller.label_events(event_start_times, returns)

    X = X.filter(items=events.index, axis=0)
    y = events["label"]
    forward_returns = events["returns"]

    return events, X, y, forward_returns

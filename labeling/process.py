from .types import EventFilter, EventLabeller, EventsDataFrame
from data_loader.types import ForwardReturnSeries, XDataFrame, ReturnSeries, ySeries
from .labellers.utils import purge_overlapping_events


def label_data(
    event_filter: EventFilter,
    event_labeller: EventLabeller,
    X: XDataFrame,
    returns: ReturnSeries,
    remove_overlapping_events: bool,
) -> tuple[EventsDataFrame, XDataFrame, ySeries, ForwardReturnSeries]:

    event_start_times = event_filter.get_event_start_times(returns)
    print(
        "| Filtered out ",
        (1 - (len(event_start_times) / len(returns))) * 100,
        "% of timestamps",
    )

    events = event_labeller.label_events(event_start_times, returns)
    if remove_overlapping_events:
        events = purge_overlapping_events(events)
        print(
            "| Purged ",
            (1 - (len(events) / len(event_start_times))) * 100,
            "% of overlapping events",
        )

    X = X.loc[events.index]
    y = events["label"]
    forward_returns = events["returns"]

    return events, X, y, forward_returns

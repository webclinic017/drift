from .event_filters.nofilter import NoEventFilter
from .event_filters.cusum import CUSUMVolatilityEventFilter, CUSUMFixedEventFilter

eventfilters_map = dict(
    none=NoEventFilter,
    cusum_vol=CUSUMVolatilityEventFilter,
    cusum_fixed=CUSUMFixedEventFilter,
)

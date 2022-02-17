from .event_filters.nofilter import NoEventFilter
from .event_filters.cusum import CUSUMVolatilityEventFilter, CUSUMFixedEventFilter

eventfilters_map = dict(
    none=NoEventFilter(),
    cusum_vol=CUSUMVolatilityEventFilter(vol_period=20),
    cusum_fixed=CUSUMFixedEventFilter(threshold=500),
)

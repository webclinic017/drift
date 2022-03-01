from .event_filters.nofilter import NoEventFilter
from .event_filters.cusum import CUSUMVolatilityEventFilter, CUSUMFixedEventFilter

eventfilters_map = dict(
    none=NoEventFilter(),
    cusum_vol=CUSUMVolatilityEventFilter(vol_period=100, multiplier=3.5),
    cusum_fixed=CUSUMFixedEventFilter(threshold=20),
)

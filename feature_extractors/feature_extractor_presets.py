from .types import FeatureExtractorConfig
from .feature_extractors import (
    feature_lag,
    feature_mom,
    feature_ROC,
    feature_RSI,
    feature_STOD,
    feature_STOK,
    feature_expanding_zscore,
    feature_vol,
    feature_day_of_month,
    feature_day_of_week,
    feature_month,
    feature_debug_future_lookahead,
)
from .fractional_differentiation import (
    feature_fractional_differentiation,
)

__presets = dict(
    debug_future_lookahead=[
        ("debug_future", feature_debug_future_lookahead, [1, 5, 10])
    ],
    single_mom=[("mom", feature_mom, [30])],
    single_vol=[("vol", feature_vol, [30])],
    mom=[("mom", feature_mom, [100, 300, 600, 900, 1800])],
    vol=[("vol", feature_vol, [100, 300, 600, 1800])],
    lags_up_to_5=[("lag", feature_lag, [1, 2, 3, 4, 5])],
    lags_up_to_10=[("lag", feature_lag, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])],
    date_all=[
        ("day_of_week", feature_day_of_week, [0]),
        ("day_of_month", feature_day_of_month, [0]),
        ("month", feature_month, [0]),
    ],
    date_days=[
        ("day_of_week", feature_day_of_week, [0]),
        ("day_of_month", feature_day_of_month, [0]),
    ],
    roc=[("roc", feature_ROC, [100, 300])],
    rsi=[("rsi", feature_ROC, [100, 300, 1000])],
    stod=[("stod", feature_STOD, [100, 300, 2000])],
    stok=[("stok", feature_STOK, [100, 300, 2000])],
    fracdiff=[("fracdiff", feature_fractional_differentiation, [100, 300])],
    z_score=[("z_score", feature_expanding_zscore, [100])],
)

presets = __presets | dict(
    level_1=__presets["mom"] + __presets["vol"],
    level_2=__presets["mom"]
    + __presets["vol"]
    + __presets["roc"]
    + __presets["rsi"]
    + __presets["stod"]
    + __presets["stok"],
)

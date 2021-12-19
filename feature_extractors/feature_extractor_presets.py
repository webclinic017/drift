from feature_extractors.feature_extractors import feature_lag, feature_mom, feature_ROC, feature_RSI, feature_STOD, feature_STOK, feature_vol, feature_day_of_month, feature_day_of_week, feature_month

lags = [('lag', feature_lag, [1,2,3,4,5,6,7,8,9])]

only_mom = [('mom', feature_mom, [30])]

date = [
    ('day_of_week', feature_day_of_week, [0]),
    ('day_of_month', feature_day_of_month, [0]),
    ('month', feature_month, [0])]

level1 = [
    ('mom', feature_mom, [10, 20, 30, 60, 90]),
    ('vol', feature_vol, [10, 20, 30, 60]),
]

level2 = level1 + [
    ('roc', feature_ROC, [10, 30]),
    ('rsi', feature_RSI, [10, 30, 100]),
    ('stod', feature_STOD, [10, 30, 200]),
    ('stok', feature_STOK, [10, 30, 200]),
]
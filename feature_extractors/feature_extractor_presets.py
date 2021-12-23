from feature_extractors.feature_extractors import feature_lag, feature_mom, feature_ROC, feature_RSI, feature_STOD, feature_STOK, feature_vol, feature_day_of_month, feature_day_of_week, feature_month, feature_debug_future_lookahead
from utils.typing import FeatureExtractorConfig
from utils.helpers import flatten

__presets = dict(
    debug_future_lookahead = [('debug_future', feature_debug_future_lookahead, [1])],
    single_mom = [('mom', feature_mom, [30])],
    single_vol = [('vol', feature_vol, [30])],
    mom = [('mom', feature_mom, [10, 20, 30, 60, 90])],
    vol = [('vol', feature_vol, [10, 20, 30, 60])],
    lags_up_to_5 = [('lag', feature_lag, [1,2,3,4,5])],
    lags_up_to_10 = [('lag', feature_lag, [1,2,3,4,5,6,7,8,9,10])],
    date_all = [
        ('day_of_week', feature_day_of_week, [0]),
        ('day_of_month', feature_day_of_month, [0]),
        ('month', feature_month, [0])],
    date_days = [
        ('day_of_week', feature_day_of_week, [0]),
        ('day_of_month', feature_day_of_month, [0]),
    ],
    roc = [('roc', feature_ROC, [10, 30])],
    rsi = [('rsi', feature_ROC, [10, 30, 100])],
    stod = [('stod', feature_STOD, [10, 30, 200])],
    stok = [('stok', feature_STOK, [10, 30, 200])],
)

presets = __presets | dict(
    level_1 = __presets["mom"] + __presets["vol"],
    level_2 = __presets["mom"] + __presets["vol"] + __presets["roc"] + __presets["rsi"] + __presets["stod"] + __presets["stok"],
)

def preprocess_feature_extractors_config(data_dict: dict) -> dict:
    keys = ['own_features', 'other_features']
    for key in keys:
        preset_names = data_dict[key]
        data_dict[key] = flatten([presets[preset_name] for preset_name in preset_names])
    return data_dict


# Use this if ever we want to create an independent boolean for each featureextractor
# def preprocess_feature_extractors_config(data_dict: dict) -> dict:
#     prefixes = ['own_features', 'other_features']
#     features_dict = dict()
#     for prefix in prefixes:
#         features_to_include = [key.replace(prefix + "_", "") for key, value in data_dict.items() if key.startswith(prefix) and value == True]
#         features_dict[prefix] = flatten([presets[feature_name] for feature_name in features_to_include])

#     data_dict = {k: v for k, v in data_dict.items() if not (k.startswith(prefixes[0]) or k.startswith(prefixes[1]))}
#     return (data_dict | features_dict)
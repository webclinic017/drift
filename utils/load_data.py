#%%
import pandas as pd
import os
import numpy as np
from utils.typing import FeatureExtractor
from typing import Literal

#%%

def get_crypto_assets(path: str) -> list[str]:
    return sorted([f.split('.')[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and 'USD' in f and not f.startswith('.')])

def get_etf_assets(path: str) -> list[str]:
    return sorted([f.split('.')[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and '_' not in f and not f.startswith('.')])


def load_data(path: str,
            target_asset: str,
            load_other_assets: bool,
            log_returns: bool,
            forecasting_horizon: int,
            own_features: list[tuple[str, FeatureExtractor, list[int]]],
            other_features: list[tuple[str, FeatureExtractor, list[int]]],
            index_column: Literal['date', 'int'],
            method: Literal['regression', 'classification'],
            no_of_classes: Literal['two', 'three-balanced', 'three-imbalanced'],
            narrow_format: bool = False,
            all_assets:list=[]
        ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads asset data from the specified path.
    Returns:
        - DataFrame `X` with all the training data
        - Series `y` with the target asset returns shifted by 1 day OR if it's a classification problem, the target class)
        - Series `forward_returns` with the target asset returns shifted by 1 day
    """

    
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and not f.startswith('.')]
    files = [f for f in files if load_other_assets == True or (load_other_assets == False and f.startswith(target_asset))]
    def is_target_asset(target_asset: str, file: str): return file.split('.')[0].startswith(target_asset)
    dfs = [__load_df(
        path=os.path.join(path,f),
        prefix=f.split('.')[0],
        returns='log_returns' if log_returns else 'returns',
        feature_extractors=own_features if is_target_asset(target_asset, f) else other_features,
        narrow_format=narrow_format,
    ) for f in files]
    if narrow_format:
        dfs = pd.concat(dfs, axis=0).fillna(0.)
    else:
        dfs = pd.concat(dfs, axis=1).fillna(0.)

    dfs.index = pd.DatetimeIndex(dfs.index)

    if index_column == 'int':
        dfs.reset_index(drop=True, inplace=True)

    if narrow_format:
        dfs = dfs.drop(index=dfs.index[0], axis=0)

    ## Create target 
    target_col = 'target'
    returns_col = target_asset + '_returns'
    forward_returns = __create_target_cum_forward_returns(dfs, returns_col, forecasting_horizon)
    if method == 'regression':
        dfs[target_col] = forward_returns
    elif method == 'classification':
        dfs[target_col] = __create_target_classes(dfs, returns_col, forecasting_horizon, no_of_classes)
    # we need to drop the last row, because we forward-shift the target (see what happens if you call .shift[-1] on a pd.Series) 
    dfs = dfs.iloc[:-forecasting_horizon]
    forward_returns = forward_returns.iloc[:-forecasting_horizon]
    
    X = dfs.drop(columns=[target_col])
    y = dfs[target_col]

    return X, y, forward_returns


def __load_df(path: str,
            prefix: str,
            returns: Literal['price', 'returns', 'log_returns'],
            feature_extractors: list[tuple[str, FeatureExtractor, list[int]]],
            narrow_format: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, index_col=0).fillna(0)

    if returns == 'log_returns':
        df['returns'] = np.log(df['close']).diff(1)
    elif returns == 'price':
        df['returns'] = df['close']
    else:
        df['returns'] = df['close'].pct_change()

    df = __apply_feature_extractors(df, log_returns=True if returns == 'log_returns' else False, feature_extractors = feature_extractors)

    df = df.replace([np.inf, -np.inf], 0.)
    df = df.drop(columns=['open', 'high', 'low', 'close'])
    # we're not ready for this just yet
    if 'volume' in df.columns:
        df = df.drop(columns=['volume'])
    
    if narrow_format:
        df["ticker"] = np.repeat(prefix, df.shape[0])
    else: 
        df.columns = [prefix + "_" + c if 'date' not in c else c for c in df.columns]
    return df


def __apply_feature_extractors(df: pd.DataFrame,
                            log_returns: bool,
                            feature_extractors: list[tuple[str, FeatureExtractor, list[int]]]) -> pd.DataFrame:

    for name, extractor, periods in feature_extractors:
        for period in periods:
            features = extractor(df, period, log_returns)
            if type(features) == pd.DataFrame:
                df = pd.concat([df, features], axis=1)
            elif type(features) == pd.Series:
                df[name + '_' + str(period)] = extractor(df, period, log_returns)
            else:
                assert False, "Feature extractor must return a pd.DataFrame or pd.Series"
    return df


# %%
def __create_target_cum_forward_returns(df: pd.DataFrame, source_column: str, period: int) -> pd.Series:
    assert period > 0
    return df[source_column].shift(-period)


def __create_target_classes(df: pd.DataFrame, source_column: str, period: int, no_of_classes: Literal["two", "three"]) -> pd.Series:
    assert period > 0

    def get_class_binary(x: float) -> int:
        return -1 if x <= 0.0 else 1

    def get_class_threeway_balanced(series: pd.Series) -> pd.Series:

        def get_bins_threeway(x):
            bins = pd.qcut(df[source_column], 3, retbins=True, duplicates = 'drop')[1]
            
            if len(bins) != 4:
                # if we don't have enough data for the quantiles, we'll need to add hard-coded values
                lower_bound = bins[0]
                upper_bound = bins[-1]
                bins = [lower_bound] + [-0.02, 0.02] + [upper_bound]
            return bins
        bins = get_bins_threeway(series)

        def map_class_threeway(current_value):
            lower_threshold = bins[1]
            upper_threshold = bins[2]
            if current_value <= lower_threshold:
                return -1
            elif current_value > lower_threshold and current_value < upper_threshold:
                return 0
            else:
                return 1
        return series.map(map_class_threeway)

    def get_class_threeway_imbalanced(series: pd.Series) -> pd.Series:

        def get_bins_threeway(x):
            bins = pd.qcut(df[source_column], 4, retbins=True, duplicates = 'drop')[1]
            
            if len(bins) != 5:
                # if we don't have enough data for the quantiles, we'll need to add hard-coded values
                lower_bound = bins[0]
                upper_bound = bins[-1]
                bins = [lower_bound] + [-0.02, 0.0, 0.02] + [upper_bound]
            return bins
        bins = get_bins_threeway(series)

        def map_class_threeway(current_value):
            lower_threshold = bins[1]
            upper_threshold = bins[3]
            if current_value <= lower_threshold:
                return -1
            elif current_value > lower_threshold and current_value < upper_threshold:
                return 0
            else:
                return 1
        return series.map(map_class_threeway)

    target_column = df[source_column].shift(-period)
    
    if no_of_classes == "three-balanced":
        return get_class_threeway_balanced(target_column)
    elif no_of_classes == "three-imbalanced":
        return get_class_threeway_imbalanced(target_column)
    else:
        return target_column.map(get_class_binary)




# These are needed for the portfolio feature, maybe we can do this in a more elegant way
# def load_crypto_only_returns(path: str, index_column: Literal['date', 'int'], returns: Literal['price', 'returns']) -> pd.DataFrame:
#     files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and 'USD' in f and not f.startswith('.')]
#     dfs = [__load_df(
#         path=os.path.join(path,f),
#         prefix=f.split('.')[0],
#         returns=returns,
#         feature_extractors=[],
#         narrow_format=False,
#     ) for f in files]
#     dfs = pd.concat(dfs, axis=1)
#     dfs = dfs.applymap(lambda x: np.nan if x == 0 else x)
#     dfs.index = pd.DatetimeIndex(dfs.index)
#     dfs.columns = [column.split('_')[0] for column in dfs.columns]
#     if index_column == 'int':
#         dfs.reset_index(drop=True, inplace=True)

#     return dfs

# def load_crypto_assets_availability(path: str, index_column: Literal['date', 'int']) -> pd.DataFrame:
#     return load_crypto_only_returns(path, index_column, 'returns').applymap(lambda x: 0 if x == 0.0 or x == 0 or np.isnan(x) else 1)


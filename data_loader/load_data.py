import pandas as pd
import numpy as np
from utils.types import DataSource, FeatureExtractor
from utils.helpers import deduplicate_indexes
from data_loader.collections import DataCollection
from typing import Literal
import ray
import os

def load_data(assets: DataCollection,
            other_assets: DataCollection,
            # exogenous_data: DataCollection,
            target_asset: str,
            load_non_target_asset: bool,
            log_returns: bool,
            forecasting_horizon: int,
            own_features: list[tuple[str, FeatureExtractor, list[int]]],
            other_features: list[tuple[str, FeatureExtractor, list[int]]],
            index_column: Literal['date', 'int'],
            method: Literal['regression', 'classification'],
            no_of_classes: Literal['two', 'three-balanced', 'three-imbalanced'],
            narrow_format: bool = False
        ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads asset data from the specified path.
    Returns:
        - DataFrame `X` with all the training data
        - Series `y` with the target asset returns shifted by 1 day OR if it's a classification problem, the target class)
        - Series `forward_returns` with the target asset returns shifted by 1 day
    """

    target_file = [f for f in assets if f[1].startswith(target_asset)]
    other_files = [f for f in assets if load_non_target_asset == True and f[1].startswith(target_asset) == False]
    files = target_file + other_files + other_assets
    def is_target_asset(target_asset: str, file: str): return file.split('.')[0].startswith(target_asset)
    futures = [__load_df.remote(
        data_source=data_source,
        prefix=data_source[1],
        returns='log_returns' if log_returns else 'returns',
        feature_extractors=own_features if is_target_asset(target_asset[1], data_source[1]) else other_features,
        narrow_format=narrow_format,
    ) for data_source in files]
    dfs = ray.get(futures)

    dfs = [deduplicate_indexes(df) for df in dfs]
    longest_df = max(dfs, key=lambda df: df.shape[0])
    if narrow_format:
        dfs = pd.concat(dfs, axis=0).fillna(0.)
    else:
        dfs = pd.concat([df.reindex(longest_df.index) for df in dfs], axis=1).fillna(0.)

    dfs.index = pd.DatetimeIndex(dfs.index)

    if index_column == 'int':
        dfs.reset_index(drop=True, inplace=True)

    if narrow_format:
        dfs = dfs.drop(index=dfs.index[0], axis=0)

    ## Create target 
    target_col = 'target'
    returns_col = target_asset[1] + '_returns'
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


@ray.remote
def __load_df(data_source: DataSource,
            prefix: str,
            returns: Literal['price', 'returns', 'log_returns'],
            feature_extractors: list[tuple[str, FeatureExtractor, list[int]]],
            narrow_format: bool = False) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_source[0], data_source[1] + '.csv'), header=0, index_col=0).fillna(0)

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


def datasource_to_file(data_source: DataSource) -> str:
    return data_source[0] + '/' + data_source[1] + '.csv'


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


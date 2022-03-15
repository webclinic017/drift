import os
import pandas as pd
import numpy as np
from .types import DataSource
from feature_extractors.types import FeatureExtractor
from utils.helpers import drop_columns_if_exist
from data_loader.collections import DataCollection
from typing import Literal, Optional
from utils.resample import upsample
from config.hashing import hash_data_config
from .types import XDataFrame, ReturnSeries
from diskcache import Cache

cache = Cache(".cachedir/data")


def load_data(**kwargs) -> tuple[XDataFrame, ReturnSeries]:
    hashed = hash_data_config(kwargs)
    if hashed in cache:
        return cache.get(hashed)
    else:
        return_value = __load_data(**kwargs)
        cache[hashed] = return_value
        return return_value


def __load_data(
    assets: DataCollection,
    other_assets: DataCollection,
    exogenous_data: DataCollection,
    target_asset: DataSource,
    load_non_target_asset: bool,
    own_features: list[tuple[str, FeatureExtractor, list[int]]],
    other_features: list[tuple[str, FeatureExtractor, list[int]]],
    exogenous_features: list[tuple[str, FeatureExtractor, list[int]]],
) -> tuple[XDataFrame, ReturnSeries]:
    """
    Loads asset data from the specified path.
    Returns:
        - DataFrame `X` with all the training data
        - Series `returns` with only the returns
        - Series `forward_returns` with the target asset returns shifted by 1 day
    """

    target_file = [f for f in assets if f.file_name.startswith(target_asset.file_name)]
    assert len(target_file) == 1, "There should be exactly one target file"
    target_freq = target_file[0].freq
    other_files = [
        f
        for f in assets
        if load_non_target_asset == True
        and f.file_name.startswith(target_asset.file_name) == False
    ]
    files = other_files + other_assets

    target_asset_df = [
        __load_df(
            data_source=data_source,
            prefix=data_source.file_name,
            returns="log_returns",
            feature_extractors=own_features,
            resample_to_freq=None,
        )
        for data_source in target_file
    ]

    df_target_asset_only_returns = __load_df(
        data_source=target_file[0],
        prefix=target_file[0].file_name,
        returns="returns",
        feature_extractors=[],
        resample_to_freq=None,
    )

    asset_dfs = [
        __load_df(
            data_source=data_source,
            prefix=data_source.file_name,
            returns="log_returns",
            feature_extractors=other_features,
            resample_to_freq=target_freq,
        )
        for data_source in files
    ]

    exogenous_dfs = [
        __load_df(
            data_source=data_source,
            prefix=data_source.file_name,
            returns="none",
            feature_extractors=exogenous_features,
            resample_to_freq=target_freq,
        )
        for data_source in exogenous_data
    ]

    X = target_asset_df + asset_dfs + exogenous_dfs
    X = pd.concat([df.reindex(X[0].index) for df in X], axis=1).fillna(0.0)

    X.index = pd.DatetimeIndex(X.index)

    ## Create target
    returns = df_target_asset_only_returns[target_asset.file_name + "_returns"]
    returns.index = pd.DatetimeIndex(X.index)

    return X, returns


def __load_df(
    data_source: DataSource,
    prefix: str,
    returns: Literal["none", "price", "returns", "log_returns"],
    feature_extractors: list[tuple[str, FeatureExtractor, list[int]]],
    resample_to_freq: Optional[Literal["5m", "1h", "1d"]] = None,
) -> pd.DataFrame:
    csv_file = os.path.join(data_source.path, data_source.file_name + ".csv")
    parquet_file = os.path.join(data_source.path, data_source.file_name + ".parquet")
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file, header=0, index_col=0).fillna(0)
    elif os.path.isfile(parquet_file):
        df = pd.read_parquet(parquet_file).fillna(0)
    else:
        raise Exception("File not found: " + data_source.path + data_source.file_name)

    if returns == "log_returns":
        df["returns"] = np.log(df["close"]).diff(1)
    elif returns == "price":
        df["returns"] = df["close"]
    elif returns == "returns":
        df["returns"] = df["close"].pct_change()

    df = __apply_feature_extractors(df, feature_extractors=feature_extractors)

    df = df.replace([np.inf, -np.inf], 0.0)
    df.index = pd.DatetimeIndex(df.index)
    df.sort_index(inplace=True)

    if resample_to_freq is not None and resample_to_freq != data_source.freq:
        df = upsample(df, resample_to_freq)

    df = drop_columns_if_exist(df, ["open", "high", "low", "close", "volume"])

    df.columns = [prefix + "_" + c if "date" not in c else c for c in df.columns]
    return df


def __apply_feature_extractors(
    df: pd.DataFrame, feature_extractors: list[tuple[str, FeatureExtractor, list[int]]]
) -> pd.DataFrame:

    for name, extractor, periods in feature_extractors:
        for period in periods:
            features = extractor(df, period)
            if type(features) == pd.DataFrame:
                df = pd.concat([df, features], axis=1)
            elif type(features) == pd.Series:
                df[name + "_" + str(period)] = features
            else:
                assert (
                    False
                ), "Feature extractor must return a pd.DataFrame or pd.Series"
    return df


def load_only_returns(
    assets: DataCollection, returns: Literal["price", "returns"]
) -> pd.DataFrame:

    assets_future = [
        __load_df(
            data_source=data_source,
            prefix=data_source.file_name,
            returns=returns,
            feature_extractors=[],
        )
        for data_source in assets
    ]
    dfs = assets_future

    dfs = pd.concat(dfs, axis=1)
    dfs.index = pd.DatetimeIndex(dfs.index)

    return dfs

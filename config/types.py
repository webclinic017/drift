from pydantic import BaseModel
from typing import Literal, Optional
from labeling.types import EventFilter
from models.base import Model
from data_loader.types import DataCollection, DataSource
from feature_extractors.types import FeatureExtractor, ScalerTypes
from labeling.types import EventFilter, EventLabeller


# RawConfig is needed to ensure we can declare config presets here with static typing, we then convert it to Config
class RawConfig(BaseModel):
    directional_models_meta: bool
    dimensionality_reduction: bool
    n_features_to_select: int
    expanding_window_base: bool
    expanding_window_meta: bool
    sliding_window_size_base: int
    sliding_window_size_meta: int
    retrain_every: int
    scaler: Literal['normalize', 'minmax', 'standardize']

    assets: list[str]
    target_asset: str
    other_assets: list[str]
    exogenous_data: list[str]
    load_non_target_asset: bool
    own_features: list[str]
    other_features: list[str]
    exogenous_features: list[str]
    event_filter: Literal['none', 'cusum_vol', 'cusum_fixed']
    labeling: Literal['two_class', 'three_class_balanced', 'three_class_imbalanced']

    directional_models: list[str]
    meta_models: list[str]


class Config(BaseModel):
    directional_models_meta: bool
    dimensionality_reduction: bool
    n_features_to_select: int
    expanding_window_base: bool
    expanding_window_meta: bool
    sliding_window_size_base: int
    sliding_window_size_meta: int
    retrain_every: int
    scaler: Literal['normalize', 'minmax', 'standardize']

    assets: DataCollection
    target_asset: DataSource
    other_assets: DataCollection
    exogenous_data: DataCollection
    load_non_target_asset: bool
    own_features: list[tuple[str, FeatureExtractor, list[int]]]
    other_features: list[tuple[str, FeatureExtractor, list[int]]]
    exogenous_features: list[tuple[str, FeatureExtractor, list[int]]]
    event_filter: EventFilter
    labeling: EventLabeller
    no_of_classes: Literal['two', 'three-balanced', 'three-imbalanced']

    mode: Literal['training', 'inference']

    directional_models: list[Model]
    meta_models: list[Model]

    class Config:
        arbitrary_types_allowed = True



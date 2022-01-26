from pydantic import BaseModel
from typing import Literal, Optional
from labeling.types import EventFilter
from models.base import Model
from data_loader.types import DataCollection, DataSource
from feature_extractors.types import FeatureExtractor, ScalerTypes
from labeling.types import EventFilter, EventLabeller


# RawConfig is needed to ensure we can declare config presets here with static typing, we then convert it to Config
class RawConfig(BaseModel):
    primary_models_meta_labeling: bool
    dimensionality_reduction: bool
    n_features_to_select: int
    expanding_window_base: bool
    expanding_window_meta_labeling: bool
    sliding_window_size_base: int
    sliding_window_size_meta_labeling: int
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

    primary_models: list[str]
    meta_labeling_models: list[str]
    ensemble_model: Optional[str]


class Config(BaseModel):
    primary_models_meta_labeling: bool
    dimensionality_reduction: bool
    n_features_to_select: int
    expanding_window_base: bool
    expanding_window_meta_labeling: bool
    sliding_window_size_base: int
    sliding_window_size_meta_labeling: int
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

    primary_models: list[tuple[str, Model]]
    meta_labeling_models: list[tuple[str, Model]]
    ensemble_model: Optional[tuple[str, Model]]

    class Config:
        arbitrary_types_allowed = True



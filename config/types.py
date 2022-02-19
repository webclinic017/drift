from pydantic import BaseModel, validator
from typing import Literal, Optional
from labeling.types import EventFilter
from models.base import Model
from data_loader.types import DataCollection, DataSource
from feature_extractors.types import FeatureExtractor
from labeling.types import EventFilter, EventLabeller
from sklearn.base import BaseEstimator
from dataclasses import dataclass
from transformations.base import Transformation

# RawConfig is needed to ensure we can declare config presets here with static typing, we then convert it to Config
class RawConfig(BaseModel):
    dimensionality_reduction_ratio: float
    n_features_to_select: int
    sliding_window_size: int
    retrain_every: int
    scaler: Literal["normalize", "minmax", "standardize", "robust"]

    assets: list[str]
    target_asset: str
    other_assets: list[str]
    exogenous_data: list[str]
    load_non_target_asset: bool
    own_features: list[str]
    other_features: list[str]
    exogenous_features: list[str]
    event_filter: Literal["none", "cusum_vol", "cusum_fixed"]
    labeling: Literal["two_class", "three_class_balanced", "three_class_imbalanced"]
    forecasting_horizon: int
    save_models: bool
    ensembling_method: Literal["voting_soft", "stacking"]

    directional_models: list[str]
    meta_models: list[str]


@dataclass
class Config:
    sliding_window_size: int
    retrain_every: int

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
    forecasting_horizon: int
    no_of_classes: Literal["two", "three-balanced", "three-imbalanced"]
    save_models: bool

    mode: Literal["training", "inference"]

    directional_model: Model
    meta_model: Model

    transformations: list[Transformation]

    @validator("directional_model", "meta_model")
    def check_model(cls, v):
        assert isinstance(v, BaseEstimator)
        return v

    @validator("event_filter")
    def check_event_filter(cls, v):
        assert isinstance(v, EventFilter)
        return v

    @validator("labeling")
    def check_labeling(cls, v):
        assert isinstance(v, EventLabeller)
        return v

    # class Config:
    #     arbitrary_types_allowed = True

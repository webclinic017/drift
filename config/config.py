
from pydantic import BaseModel
from typing import Literal, Optional
from models.base import Model
from utils.types import DataCollection, DataSource, FeatureExtractor

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
    log_returns: bool
    forecasting_horizon: int
    own_features: list[str]
    other_features: list[str]
    exogenous_features: list[str]
    no_of_classes: Literal['two', 'three-balanced', 'three-imbalanced']

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
    log_returns: bool
    forecasting_horizon: int
    own_features: list[tuple[str, FeatureExtractor, list[int]]]
    other_features: list[tuple[str, FeatureExtractor, list[int]]]
    exogenous_features: list[tuple[str, FeatureExtractor, list[int]]]
    no_of_classes: Literal['two', 'three-balanced', 'three-imbalanced']

    primary_models: list[tuple[str, Model]]
    meta_labeling_models: list[tuple[str, Model]]
    ensemble_model: Optional[tuple[str, Model]]

    class Config:
        arbitrary_types_allowed = True


def get_dev_config() -> RawConfig:
  
    regression_models = ["Lasso"]
    classification_models = ["LogisticRegression_two_class"]

    return RawConfig(
        primary_models_meta_labeling = False,
        dimensionality_reduction = False,
        n_features_to_select = 30,
        expanding_window_base = False,
        expanding_window_meta_labeling = False,
        sliding_window_size_base = 380,
        sliding_window_size_meta_labeling = 1,
        retrain_every = 20,
        scaler = 'minmax', # 'normalize' 'minmax' 'standardize'

        assets = ['daily_only_btc'],
        target_asset = 'BTC_USD',
        other_assets = [],
        exogenous_data = [],
        load_non_target_asset= True,
        log_returns= True,
        forecasting_horizon = 1,
        own_features = ['level_2', 'date_days'],
        other_features = ['single_mom'],
        exogenous_features = ['z_score'],
        no_of_classes= 'two',

        primary_models = classification_models,
        meta_labeling_models = [],
        ensemble_model = None
    )


def get_default_ensemble_config() -> RawConfig:
  
    regression_models = ["Lasso", "KNN", "RFR"]
    classification_models = ["LogisticRegression_two_class", "LDA", "NB", "RFC", "XGB_two_class", "LGBM", "StaticMom"]
    meta_labeling_models = ['LogisticRegression_two_class', 'LGBM']
    ensemble_model = 'Average'

    return RawConfig(
        primary_models_meta_labeling = True,
        dimensionality_reduction = False,
        n_features_to_select = 30,
        expanding_window_base = False,
        expanding_window_meta_labeling = True,
        sliding_window_size_base = 380,
        sliding_window_size_meta_labeling = 240,
        retrain_every = 10,
        scaler = 'minmax', # 'normalize' 'minmax' 'standardize'

        assets = ['daily_crypto'],
        target_asset = 'BTC_USD',
        other_assets = ['daily_etf'],
        exogenous_data = ['daily_glassnode'],
        load_non_target_asset= True,
        log_returns= True,
        forecasting_horizon = 1,
        own_features = ['level_2', 'date_days', 'lags_up_to_5'],
        other_features = ['level_2', 'lags_up_to_5'],
        exogenous_features = ['z_score'],
        no_of_classes= 'two',

        primary_models = classification_models,
        meta_labeling_models = meta_labeling_models,
        ensemble_model = ensemble_model
    )



def get_lightweight_ensemble_config() -> RawConfig:
  
    regression_models = ["Lasso", "KNN"]
    classification_models = ['LogisticRegression_two_class', 'SVC']
    meta_labeling_models = ['LogisticRegression_two_class', 'LGBM']
    ensemble_model = 'Average'

    return RawConfig(
        primary_models_meta_labeling = True,
        dimensionality_reduction = True,
        n_features_to_select = 30,
        expanding_window_base = False,
        expanding_window_meta_labeling = True,
        sliding_window_size_base = 380,
        sliding_window_size_meta_labeling = 240,
        retrain_every = 40,
        scaler = 'minmax', # 'normalize' 'minmax' 'standardize'

        assets = ['daily_crypto_lightweight'],
        target_asset = 'BCH_USD',
        other_assets = ['daily_etf'],
        exogenous_data = ['daily_glassnode'],
        load_non_target_asset= True,
        log_returns= True,
        forecasting_horizon = 1,
        own_features = ['level_2' ],
        other_features = ['level_2'],
        exogenous_features = ['z_score'],
        no_of_classes= 'two',

        primary_models = classification_models,
        meta_labeling_models = meta_labeling_models,
        ensemble_model = ensemble_model
    )



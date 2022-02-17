from .types import RawConfig, Config


def get_dev_config() -> RawConfig:

    classification_models = ["LogisticRegression_two_class"]

    return RawConfig(
        directional_models_meta=False,
        dimensionality_reduction=False,
        n_features_to_select=30,
        expanding_window_base=False,
        expanding_window_meta=False,
        sliding_window_size_base=380,
        sliding_window_size_meta=1,
        retrain_every=20,
        scaler="minmax",  # 'normalize' 'minmax' 'standardize'
        assets=["daily_only_btc"],
        target_asset="BTC_USD",
        other_assets=[],
        exogenous_data=[],
        load_non_target_asset=True,
        own_features=["level_2", "date_days"],
        other_features=["single_mom"],
        exogenous_features=["z_score"],
        directional_models=classification_models,
        meta_models=[],
        event_filter="none",
        labeling="two_class",
        forecasting_horizon=100,
    )


def get_default_ensemble_config() -> RawConfig:

    classification_models = [
        "LogisticRegression_two_class",
        "LDA",
        "NB",
        "RFC",
        "XGB_two_class",
        "LGBM",
        "StaticMom",
    ]
    meta_models = ["LogisticRegression_two_class", "LGBM"]

    return RawConfig(
        directional_models_meta=True,
        dimensionality_reduction=False,
        n_features_to_select=30,
        expanding_window_base=False,
        expanding_window_meta=True,
        sliding_window_size_base=380,
        sliding_window_size_meta=240,
        retrain_every=10,
        scaler="minmax",  # 'normalize' 'minmax' 'standardize'
        assets=["daily_crypto"],
        target_asset="BTC_USD",
        other_assets=["daily_etf"],
        exogenous_data=["daily_glassnode"],
        load_non_target_asset=True,
        own_features=["level_2", "date_days", "lags_up_to_5"],
        other_features=["level_2", "lags_up_to_5"],
        exogenous_features=["z_score"],
        directional_models=classification_models,
        meta_models=meta_models,
        event_filter="cusum_vol",
        labeling="two_class",
        forecasting_horizon=100,
    )


def get_lightweight_ensemble_config() -> RawConfig:

    classification_models = ["LogisticRegression_two_class", "LGBM"]
    meta_models = ["LogisticRegression_two_class", "LGBM"]

    return RawConfig(
        directional_models_meta=True,
        dimensionality_reduction=True,
        n_features_to_select=30,
        expanding_window_base=True,
        expanding_window_meta=True,
        sliding_window_size_base=3800,
        sliding_window_size_meta=2400,
        retrain_every=1000,
        scaler="minmax",  # 'normalize' 'minmax' 'standardize'
        assets=["fivemin_crypto"],
        target_asset="BTC_USD",
        other_assets=[],
        exogenous_data=[],
        load_non_target_asset=False,
        own_features=["level_1"],
        other_features=[],
        exogenous_features=[],
        directional_models=classification_models,
        meta_models=meta_models,
        event_filter="cusum_fixed",
        labeling="two_class",
        forecasting_horizon=50,
    )

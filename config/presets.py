from .types import RawConfig, Config


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
        dimensionality_reduction_ratio=0.5,
        n_features_to_select=30,
        sliding_window_size=380,
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

    classification_models = [
        "LogisticRegression_two_class",
        "LDA",
        "NB",
        "RFC",
        "LGBM",
        # "StaticMom",
    ]
    meta_models = ["LogisticRegression_two_class", "LGBM"]

    return RawConfig(
        dimensionality_reduction_ratio=0.5,
        n_features_to_select=30,
        sliding_window_size=3800,
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

from .types import RawConfig


def get_default_config() -> RawConfig:

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
        start_date=None,
        dimensionality_reduction_ratio=0.5,
        n_features_to_select=50,
        initial_window_size=3800,
        retrain_every=2000,
        scaler="minmax",  # 'normalize' 'minmax' 'standardize' 'robust'
        assets=["fivemin_crypto"],
        target_asset="BTCUSDT",
        other_assets=[],
        exogenous_data=[],
        load_non_target_asset=True,
        own_features=["level_1"],
        other_features=["z_score"],
        exogenous_features=[],
        directional_models=classification_models,
        meta_models=meta_models,
        event_filter="cusum_vol",
        event_filter_multiplier=3.5,
        remove_overlapping_events=True,
        labeling="two_class",
        forecasting_horizon=10,
        transaction_costs=0.002,
        save_models=True,
        ensembling_method="voting_soft",
    )


def get_minimal_config() -> RawConfig:

    classification_models = [
        "LogisticRegression_two_class",
        # "LDA",
        # "NB",
        # "RFC",
        # "LGBM",
        # "StaticMom",
    ]
    meta_models = ["LogisticRegression_two_class", "LGBM"]

    return RawConfig(
        dimensionality_reduction_ratio=0,
        n_features_to_select=0,
        initial_window_size=3800,
        retrain_every=2000,
        scaler="minmax",  # 'normalize' 'minmax' 'standardize' 'robust'
        assets=["fivemin_crypto"],
        target_asset="BTCUSDT",
        other_assets=[],
        exogenous_data=[],
        load_non_target_asset=False,
        own_features=[],
        other_features=[],
        exogenous_features=[],
        directional_models=classification_models,
        meta_models=meta_models,
        event_filter="none",
        event_filter_multiplier=3.5,
        remove_overlapping_events=True,
        labeling="two_class",
        forecasting_horizon=10,
        transaction_costs=0.002,
        save_models=True,
        ensembling_method="voting_soft",
    )

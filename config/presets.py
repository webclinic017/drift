from .types import RawConfig, Config


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
        dimensionality_reduction_ratio=0.5,
        n_features_to_select=50,
        sliding_window_size=3800,
        retrain_every=2000,
        scaler="minmax",  # 'normalize' 'minmax' 'standardize' 'robust'
        assets=["fivemin_crypto"],
        target_asset="BTCUSDT",
        other_assets=[],
        exogenous_data=[],
        load_non_target_asset=True,
        own_features=["level_2"],
        other_features=["z_score"],
        exogenous_features=[],
        directional_models=classification_models,
        meta_models=meta_models,
        event_filter="cusum_vol",
        remove_overlapping_events=False,
        labeling="two_class",
        forecasting_horizon=10,
        transaction_costs=0.002,
        save_models=True,
        ensembling_method="voting_soft",
    )

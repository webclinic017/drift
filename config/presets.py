from .types import RawConfig, Config

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
        own_features = ['level_2', 'date_days'],
        other_features = ['single_mom'],
        exogenous_features = ['z_score'],

        primary_models = classification_models,
        meta_labeling_models = [],
        ensemble_model = None,

        event_filter = 'none',
        labeling = 'two_class'
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
        own_features = ['level_2', 'date_days', 'lags_up_to_5'],
        other_features = ['level_2', 'lags_up_to_5'],
        exogenous_features = ['z_score'],

        primary_models = classification_models,
        meta_labeling_models = meta_labeling_models,
        ensemble_model = ensemble_model,

        event_filter = 'cusum_vol',
        labeling = 'two_class'
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
        own_features = ['level_2' ],
        other_features = ['level_2'],
        exogenous_features = ['z_score'],

        primary_models = classification_models,
        meta_labeling_models = meta_labeling_models,
        ensemble_model = ensemble_model,

        event_filter = 'none',
        labeling = 'two_class'
    )



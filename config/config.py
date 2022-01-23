
def get_dev_config() -> tuple[dict, dict, dict]:
  
    training_config = dict(
        primary_models_meta_labeling = False,
        dimensionality_reduction = False,
        n_features_to_select = 30,
        expanding_window_primary = False,
        expanding_window_meta_labeling = False,
        sliding_window_size_primary = 380,
        sliding_window_size_meta_labeling = 1,
        retrain_every = 20,
        scaler = 'minmax', # 'normalize' 'minmax' 'standardize'
    )

    data_config = dict(
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
        index_column= 'int',
        no_of_classes= 'two',
        narrow_format = False,
    )

    regression_models = ["Lasso"]
    classification_models = ["LogisticRegression_two_class"]

    model_config = dict(
        primary_models = classification_models,
        meta_labeling_models = [],
        ensemble_model = None
    )
    
    return model_config, training_config, data_config



def get_default_ensemble_config() -> tuple[dict, dict, dict]:
  
    training_config = dict(
        primary_models_meta_labeling = True,
        dimensionality_reduction = False,
        n_features_to_select = 30,
        expanding_window_primary = False,
        expanding_window_meta_labeling = True,
        sliding_window_size_primary = 380,
        sliding_window_size_meta_labeling = 240,
        retrain_every = 10,
        scaler = 'minmax', # 'normalize' 'minmax' 'standardize'
    )

    data_config = dict(
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
        index_column= 'int',
        no_of_classes= 'two',
        narrow_format = False,
    )

    regression_models = ["Lasso", "KNN", "RFR"]
    classification_models = ["LogisticRegression_two_class", "LDA", "NB", "RFC", "XGB_two_class", "LGBM", "StaticMom"]
    meta_labeling_models = ['LogisticRegression_two_class', 'LGBM']
    ensemble_model = 'Average'

    model_config = dict(
        primary_models = classification_models,
        meta_labeling_models = meta_labeling_models,
        ensemble_model = ensemble_model
    )
    
    return model_config, training_config, data_config



def get_lightweight_ensemble_config() -> tuple[dict, dict, dict]:
  
    training_config = dict(
        primary_models_meta_labeling = True,
        dimensionality_reduction = True,
        n_features_to_select = 30,
        expanding_window_primary = False,
        expanding_window_meta_labeling = True,
        sliding_window_size_primary = 380,
        sliding_window_size_meta_labeling = 240,
        retrain_every = 40,
        scaler = 'minmax', # 'normalize' 'minmax' 'standardize'
    )

    data_config = dict(
        assets = ['daily_crypto_lightweight'],
        target_asset = 'BTC_USD',
        other_assets = ['daily_etf'],
        exogenous_data = ['daily_glassnode'],
        load_non_target_asset= True,
        log_returns= True,
        forecasting_horizon = 1,
        own_features = ['level_2' ],
        other_features = ['level_2'],
        exogenous_features = ['z_score'],
        index_column= 'int',
        no_of_classes= 'two',
        narrow_format = False,
    )

    regression_models = ["Lasso", "KNN"]
    classification_models = ['LogisticRegression_two_class', 'SVC']
    meta_labeling_models = ['LogisticRegression_two_class', 'LGBM']
    ensemble_model = 'Average'

    model_config = dict(
        primary_models = classification_models,
        meta_labeling_models = meta_labeling_models,
        ensemble_model = ensemble_model
    )
    
    return model_config, training_config, data_config



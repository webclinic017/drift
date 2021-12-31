from models.model_map import model_names_classification, model_names_regression


def get_default_level_1_daily_config() -> tuple[dict, dict, dict]:
  
    training_config = dict(
        dimensionality_reduction = True,
        feature_selection = True,
        n_features_to_select = 30,
        expanding_window_level1 = False,
        expanding_window_level2 = False,
        sliding_window_size_level1 = 380,
        sliding_window_size_level2 = 1,
        retrain_every = 20,
        scaler = 'minmax', # 'normalize' 'minmax' 'standardize' 'none'
        include_original_data_in_ensemble = False,
    )

    data_config = dict(
        assets = ['hourly_crypto'],
        other_assets = [],
        # exogenous_data = [],
        load_non_target_asset= True,
        log_returns= True,
        forecasting_horizon = 1,
        own_features = ['level_2', 'date_days'],
        other_features = ['single_mom'],
        index_column= 'int',
        method= 'classification',
        no_of_classes= 'three-balanced'
    )

    regression_models = ["Lasso"]
    classification_models = ["KNN"]

    model_config = dict(
        level_1_models = regression_models if data_config['method'] == 'regression' else classification_models,
        level_2_model = None
    )
    
    return model_config, training_config, data_config


def get_default_level_2_hourly_config() -> tuple[dict, dict, dict]:
  
    training_config = dict(
        dimensionality_reduction = True,
        feature_selection = True,
        n_features_to_select = 30,
        expanding_window_level1 = True,
        expanding_window_level2 = False,
        sliding_window_size_level1 = 2480,
        sliding_window_size_level2 = 1,
        retrain_every = 100,
        scaler = 'minmax', # 'normalize' 'minmax' 'standardize' 'none'
        include_original_data_in_ensemble = False,
    )

    data_config = dict(
        assets = ['hourly_crypto'],
        other_assets = [],
        # exogenous_data = [],
        load_non_target_asset= True,
        log_returns= True,
        forecasting_horizon = 1,
        own_features = ['level_2', 'date_days', 'lags_up_to_5'],
        other_features = ['level_2'],
        index_column= 'int',
        method= 'classification',
        no_of_classes= 'three-balanced'
    )

    regression_models = ["Lasso", "KNN", "RF"]
    regression_ensemble_model = 'KNN'
    classification_models = ["LDA", "KNN", "CART", "RF", "StaticMom"]
    classification_ensemble_model = 'Ensemble_Average'

    model_config = dict(
        level_1_models = regression_models if data_config['method'] == 'regression' else classification_models,
        level_2_model = regression_ensemble_model if data_config['method'] == 'regression' else classification_ensemble_model
    )
    
    return model_config, training_config, data_config


def get_default_level_2_daily_config() -> tuple[dict, dict, dict]:
  
    training_config = dict(
        dimensionality_reduction = True,
        feature_selection = True,
        n_features_to_select = 30,
        expanding_window_level1 = True,
        expanding_window_level2 = False,
        sliding_window_size_level1 = 380,
        sliding_window_size_level2 = 1,
        retrain_every = 20,
        scaler = 'minmax', # 'normalize' 'minmax' 'standardize' 'none'
        include_original_data_in_ensemble = False,
    )

    data_config = dict(
        assets = ['daily_crypto'],
        other_assets = ['daily_etf'],
        # exogenous_data = [],
        load_non_target_asset= True,
        log_returns= True,
        forecasting_horizon = 1,
        own_features = ['level_2', 'date_days', 'fracdiff'],
        other_features = ['level_2', 'fracdiff'],
        index_column= 'int',
        method= 'classification',
        no_of_classes= 'three-balanced'
    )

    regression_models = ["Lasso", "KNN", "RF"]
    regression_ensemble_model = 'KNN'
    classification_models = ["LDA", "KNN", "CART", "RF", "StaticMom"]
    classification_ensemble_model = 'Ensemble_Average'

    model_config = dict(
        level_1_models = regression_models if data_config['method'] == 'regression' else classification_models,
        level_2_model = regression_ensemble_model if data_config['method'] == 'regression' else classification_ensemble_model
    )
    
    return model_config, training_config, data_config


def validate_config(model_config:dict, training_config:dict, data_config:dict):
    # We need to make sure there's only one output from the pipeline
    # If level-2 model is there, we need more than one level-1 models to train
    if model_config["level_2_model"] is not None: assert len(model_config["level_1_models"]) > 0
    # If there's no level-2 model, we need to have only one level-1 model
    if model_config["level_2_model"] is None: assert len(model_config["level_1_models"]) == 1

def get_model_name(model_config:dict) -> str:
    if model_config["level_2_model"] is not None:
        return model_config["level_2_model"][0]
    elif len(model_config["level_1_models"]) == 1:
        return model_config["level_1_models"][0][0]
    else:
        raise Exception("No model name found")
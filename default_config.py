from utils.load_data import get_crypto_assets
import feature_extractors.feature_extractor_presets as feature_extractor_presets
from models.model_map import model_names_classification, model_names_regression

def get_default_config() -> tuple[dict, dict, dict]:
  
    training_config = dict(
        sliding_window_size = 150,
        retrain_every = 20,
        scaler = 'minmax', # 'normalize' 'minmax' 'standardize' 'none'
        include_original_data_in_ensemble = True,
    )

    data_config = dict(
        path='data/',
        all_assets = get_crypto_assets('data/'),
        load_other_assets= False,
        log_returns= True,
        forecasting_horizon = 1,
        own_features= feature_extractor_presets.date + feature_extractor_presets.level1,
        other_features= [],
        index_column= 'int',
        method= 'classification',
    )

    # regression_models = ["Lasso", "Ridge", "BayesianRidge", "KNN", "AB", "LR", "MLP", "RF", "SVR"]
    regression_models = model_names_regression
    regression_ensemble_models = ['Ensemble_Average']
    # classification_models = ["LR", "LDA", "KNN", "CART", "NB", "AB", "RF", "StaticMom"]
    classification_models = model_names_classification        
    classification_ensemble_models = ['Ensemble_Average']

    model_config = dict(
        level_1_models = regression_models if data_config['method'] == 'regression' else classification_models,
        level_2_models = regression_ensemble_models if data_config['method'] == 'regression' else classification_ensemble_models
    )
    
    return model_config, training_config, data_config
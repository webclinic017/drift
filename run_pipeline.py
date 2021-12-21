from sklearnex import patch_sklearn
patch_sklearn()

from utils.load_data import get_crypto_assets, get_etf_assets, load_data

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

import feature_extractors.feature_extractor_presets as feature_extractor_presets
from training.pipeline import run_single_asset_trainig_pipeline

from typing import Tuple


def get_config()->Tuple[dict, dict, dict]:

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

    classification_models = [
        ('LR', LogisticRegression(n_jobs=-1)),
        # ('LDA', LinearDiscriminantAnalysis()),
        ('KNN', KNeighborsClassifier()),
        # ('CART', DecisionTreeClassifier()),
        # ('NB', GaussianNB()),
        # ('AB', AdaBoostClassifier()),
        # ('RF', RandomForestClassifier(n_jobs=-1))
    ]

    regression_models = [
        # ('Lasso', Lasso(alpha=0.1, max_iter=1000)),
        ('Ridge', Ridge(alpha=0.1)),
        ('BayesianRidge', BayesianRidge()),
        # ('KNN', KNeighborsRegressor(n_neighbors=25)),
        # ('AB', AdaBoostRegressor(random_state=1)),
        # ('LR', LinearRegression(n_jobs=-1)),
        # ('MLP', MLPRegressor(hidden_layer_sizes=(100,20), max_iter=1000)),
        # ('RF', RandomForestRegressor(n_jobs=-1)),
        # ('SVR', SVR(kernel='rbf', C=1e3, gamma=0.1))
    ]

    regression_ensemble_model = [('Ensemble - Ridge', Ridge(alpha=0.1))]
    classification_ensemble_model = [('Ensemble - CART', DecisionTreeClassifier())]

    model_config = dict(
        level_1_models = regression_models if data_config['method'] == 'regression' else classification_models,
        level_2_model = regression_ensemble_model if data_config['method'] == 'regression' else classification_ensemble_model,
    )
    return model_config, training_config, data_config

def launch_wandb(config, sweep=False):
        from wandb_setup import get_wandb
        wandb = get_wandb()   
        
        if type(wandb) == type(None): 
            return None
        elif sweep:
            wandb.init(project="price-forecasting",  config = config)             
            return wandb
        else:
            wandb.init(project="price-forecasting", config=config, reinit=True)
            return wandb
    

def run_pipeline(with_wandb: bool, sweep: bool):
    model_config, training_config, data_config = get_config()
    
    wandb = None
    if with_wandb: 
        wandb = launch_wandb(dict(**model_config, **training_config, **data_config), sweep)
        
        if type(wandb) is not type(None):
            for k in training_config: training_config[k] = wandb.config[k]
        # for k in model_config: model_config[k] = wandb.config[k]
        # for k in data_config: data_config[k] = wandb.config[k]
            
    pipeline(model_config, training_config, data_config, wandb)  
    

# Run pipeline

def pipeline(model_config:dict, training_config:dict, data_config:dict, wandb):
    results = pd.DataFrame()

    for asset in data_config['all_assets']:
        print('--------\nPredicting: ', asset)
        all_predictions = pd.DataFrame()

        # 1. Load data
        data_params = data_config.copy()
        data_params['target_asset'] = asset

        X, y, target_returns = load_data(**data_params)

        # 2. Train Level-1 models
        current_result, current_predictions = run_single_asset_trainig_pipeline(
            ticker_to_predict = asset,
            X = X,
            y = y,
            target_returns = target_returns,
            models = model_config['level_1_models'],
            method = data_config['method'],
            sliding_window_size = training_config['sliding_window_size'],
            retrain_every =  training_config['retrain_every'],
            scaler =  training_config['scaler'],
            wandb = wandb
        )
        results = pd.concat([results, current_result], axis=1)
        all_predictions = pd.concat([all_predictions, current_predictions], axis=1)

        # 3. Train Level-2 (Ensemble) model
        ensemble_X = all_predictions
        if training_config['include_original_data_in_ensemble']:
            ensemble_X = pd.concat([ensemble_X, X], axis=1)

        ensemble_result, ensemble_preds = run_single_asset_trainig_pipeline(
            ticker_to_predict = asset,
            X = ensemble_X,
            y = y,
            target_returns = target_returns,
            models = model_config['level_2_model'],
            method = data_config['method'],
            sliding_window_size = training_config['sliding_window_size'],
            retrain_every = training_config['retrain_every'],
            scaler = training_config['scaler'],
            wandb = wandb
        )

        results = pd.concat([results, ensemble_result], axis=1)
        all_predictions = pd.concat([all_predictions, ensemble_preds], axis=1)


    results.to_csv('results.csv')

    level1_columns = results[[column for column in results.columns if 'Ensemble' not in column]]
    ensemble_columns = results[[column for column in results.columns if 'Ensemble' in column]]

    print("Mean Sharpe ratio for Level-1 models: ", level1_columns.loc['sharpe'].mean())
    print("Mean Sharpe ratio for Level-2 (Ensemble) models: ", ensemble_columns.loc['sharpe'].mean())

    
    
if __name__ == '__main__':
    run_pipeline(with_wandb = False, sweep = False)
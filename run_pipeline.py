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


# Parameters
regression_models = [
    # ('Lasso', Lasso(alpha=0.1, max_iter=1000)),
    ('Ridge', Ridge(alpha=0.1)),
    ('BayesianRidge', BayesianRidge()),
    ('KNN', KNeighborsRegressor(n_neighbors=25)),
    # ('AB', AdaBoostRegressor(random_state=1)),
    # ('LR', LinearRegression(n_jobs=-1)),
    # ('MLP', MLPRegressor(hidden_layer_sizes=(100,20), max_iter=1000)),
    # ('RF', RandomForestRegressor(n_jobs=-1)),
    # ('SVR', SVR(kernel='rbf', C=1e3, gamma=0.1))
]
regression_ensemble_model = [('Ensemble - Ridge', Ridge(alpha=0.1))]

classification_models = [
    ('LR', LogisticRegression(n_jobs=-1)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    # ('AB', AdaBoostClassifier()),
    # ('RF', RandomForestClassifier(n_jobs=-1))
]
classification_ensemble_model = [('Ensemble - CART', DecisionTreeClassifier())]


path = 'data/'
all_assets = get_crypto_assets(path)

sliding_window_size = 150
retrain_every = 20
scaler = 'minmax' # 'normalize' 'minmax' 'standardize' 'none'
include_original_data_in_ensemble = True
method = 'classification'
forecasting_horizon = 1
data_parameters = dict(path=path,
    load_other_assets= False,
    log_returns= True,
    forecasting_horizon = forecasting_horizon,
    own_features= feature_extractor_presets.date + feature_extractor_presets.level1,
    other_features= [],
    index_column= 'int',
    method= method,
)


# Run pipeline

results = pd.DataFrame()

for asset in all_assets:
    print('--------\nPredicting: ', asset)
    all_predictions = pd.DataFrame()

    # 1. Load data
    data_params = data_parameters.copy()
    data_params['target_asset'] = asset

    X, y, target_returns = load_data(**data_params)

    # 2. Train Level-1 models
    current_result, current_predictions = run_single_asset_trainig_pipeline(
        ticker_to_predict = asset,
        X = X,
        y = y,
        target_returns = target_returns,
        models = regression_models if method == 'regression' else classification_models,
        method = method,
        sliding_window_size = sliding_window_size,
        retrain_every = retrain_every,
        scaler = scaler
    )
    results = pd.concat([results, current_result], axis=1)
    all_predictions = pd.concat([all_predictions, current_predictions], axis=1)

    # 3. Train Level-2 (Ensemble) model
    ensemble_X = all_predictions
    if include_original_data_in_ensemble:
        ensemble_X = pd.concat([ensemble_X, X], axis=1)

    ensemble_result, ensemble_preds = run_single_asset_trainig_pipeline(
        ticker_to_predict = asset,
        X = ensemble_X,
        y = y,
        target_returns = target_returns,
        models = regression_ensemble_model if method == 'regression' else classification_ensemble_model,
        method = method,
        sliding_window_size = sliding_window_size,
        retrain_every = retrain_every,
        scaler = scaler
    )

    results = pd.concat([results, ensemble_result], axis=1)
    all_predictions = pd.concat([all_predictions, ensemble_preds], axis=1)

results.to_csv('results.csv')

level1_columns = results[[column for column in results.columns if 'Ensemble' not in column]]
ensemble_columns = results[[column for column in results.columns if 'Ensemble' in column]]

print("Mean Sharpe ratio for Level-1 models: ", level1_columns.loc['sharpe'].mean())
print("Mean Sharpe ratio for Level-2 (Ensemble) models: ", ensemble_columns.loc['sharpe'].mean())
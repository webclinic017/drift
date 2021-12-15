from logging import log
from typing import Literal
from sklearnex import patch_sklearn
patch_sklearn()

from load_data import get_crypto_assets, get_etf_assets, load_data
from utils.evaluate import evaluate_predictions

import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler

from utils.walk_forward import walk_forward_train_test

regression_models = [
    # ('LR', LinearRegression(n_jobs=-1)),
    ('Lasso', Lasso(alpha=0.1, max_iter=10000)),
    ('Ridge', Ridge(alpha=1.0)),
    ('BayesianRidge', BayesianRidge()),
    ('KNN', KNeighborsRegressor(n_neighbors=15)),
    # ('MLP', MLPRegressor(hidden_layer_sizes=(100,20), max_iter=1000)),
    ('AB', AdaBoostRegressor()),
    # ('RF', RandomForestRegressor(n_jobs=-1)),
    # ('SVR', SVR(kernel='rbf', C=1e3, gamma=0.1))
]
classification_models = [
    ('LR', LogisticRegression(n_jobs=-1)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('AB', AdaBoostClassifier()),
    ('RF', RandomForestClassifier(n_jobs=-1))
]



def run_whole_pipeline(
                    ticker_to_predict: str,
                    load_data_args: dict,
                    models,
                    method: Literal['regression', 'classification'],
                    sliding_window_size: int,
                    retrain_every: int,
                    scaling: bool,
                    ):
    print('--------\nPredicting: ', ticker_to_predict)

    X, y = load_data(**load_data_args)

    if scaling:
        # TODO: should move scaling to an expanding window compomenent, probably worth not turning it on for now
        feature_scaler = MinMaxScaler(feature_range= (-1, 1))
        X = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns, index=X.index)
        # TODO: should scale y as well probably

    results = pd.DataFrame()

    for model_name, model in models:

        model_over_time, preds = walk_forward_train_test(
            model_name=model_name,
            model = model,
            X = X,
            y = y,
            window_size = sliding_window_size,
            retrain_every = retrain_every
        )
        result = evaluate_predictions(
            model_name = model_name,
            y_true = y,
            y_pred = preds,
            sliding_window_size = sliding_window_size,
            method = method,
        )
        column_name = ticker_to_predict + "_" + model_name
        results[column_name] = result

    return results

results = pd.DataFrame()
all_assets = get_crypto_assets('data/')



for asset in all_assets:
    for method in ['regression']:
        load_data_args = dict(path='data/',
            target_asset= asset,
            target_asset_lags= [1,2,3,4,5,6,8,10,15],
            load_other_assets= False,
            other_asset_lags= [],
            log_returns= True,
            add_date_features= True,
            own_technical_features= 'level2',
            other_technical_features= 'none',
            exogenous_features= 'none',
            index_column= 'int',
            method= method,
        )

        current_result = run_whole_pipeline(
            ticker_to_predict = asset,
            load_data_args = load_data_args,
            models = regression_models if method == 'regression' else classification_models,
            method = method,
            sliding_window_size = 120,
            retrain_every = 50,
            scaling = False
        )
        results = pd.concat([results, current_result], axis=1)

results.to_csv('results.csv')
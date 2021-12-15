from typing import Literal
from sklearnex import patch_sklearn
patch_sklearn()

from load_data import get_all_assets, load_data
from utils.evaluate import evaluate_predictions

import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge, LogisticRegression
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
    ('LR', LinearRegression(n_jobs=-1)),
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
                    models,
                    method: Literal['regression', 'classification'],
                    sliding_window_size: int,
                    retrain_every: int,
                    scaling: bool,
                    ):
    print('--------\nPredicting: ', ticker_to_predict)

    X, y = load_data(path='data/',
        target_asset=ticker_to_predict,
        target_asset_lags=[1,2,3,4,5,6,8,10,15],
        load_other_assets=False,
        other_asset_lags=[],
        log_returns=True,
        add_date_features=True,
        own_technical_features='level2',
        other_technical_features='none',
        exogenous_features='none',
        index_column='int',
        method=method,
    )


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
        result = evaluate_predictions(model_name, y, preds, sliding_window_size, method)
        column_name = ticker_to_predict + "_" + model_name
        results[column_name] = result

    return results

results = pd.DataFrame()
all_assets = get_all_assets('data/')

for asset in all_assets:
    for method in ['regression']:
        current_result = run_whole_pipeline(
            ticker_to_predict = asset,
            models = regression_models if method == 'regression' else classification_models,
            method = method,
            sliding_window_size = 120,
            retrain_every = 50,
            scaling = False
        )
        results = pd.concat([results, current_result], axis=1)

results.to_csv('results.csv')
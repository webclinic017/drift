#%% Import all the stuff, load data, define constants
from typing import Literal
from sklearnex import patch_sklearn
patch_sklearn()

from load_data import create_target_cum_forward_returns, load_data, create_target_classes
from sktime.forecasting.model_selection import temporal_train_test_split
from utils.evaluate import evaluate_predictions_regression, evaluate_predictions_classification

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from utils.walk_forward import walk_forward_train_test

regression_models = [
    ('LR', LinearRegression(n_jobs=-1)),
    ('BayesianRidge', BayesianRidge()),
    ('KNN', KNeighborsRegressor(n_neighbors=15)),
    # ('MLP', MLPRegressor(hidden_layer_sizes=(100,20), max_iter=1000)),
    ('AB', AdaBoostRegressor()),
    # ('RF', lambda: RandomForestRegressor(n_jobs=-1)),
    ('SVR', SVR(kernel='rbf', C=1e3, gamma=0.1))
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
    print('Predicting: ', ticker_to_predict)

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
        # TODO: should move scaling to an expanding window compomenent
        feature_scaler = MinMaxScaler(feature_range= (-1, 1))
        X = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns, index=X.index)
        # TODO: should scale y as well probably

    for model_name, model in models:

        model_over_time, preds = walk_forward_train_test(
            model_name=model_name,
            model = model,
            X = X,
            y = y,
            window_size = sliding_window_size,
            retrain_every = retrain_every
        )
        if method == 'regression':
            evaluate_predictions_regression(model_name, y, preds, sliding_window_size)
        elif method == 'classification':
            evaluate_predictions_classification(model_name, y, preds, sliding_window_size)


ticker_to_predict = 'BTC_USD'
run_whole_pipeline(
    ticker_to_predict = ticker_to_predict,
    models = regression_models,
    method = 'regression',
    sliding_window_size = 120,
    retrain_every = 50,
    scaling=False
)
run_whole_pipeline(
    ticker_to_predict = ticker_to_predict,
    models = classification_models,
    method = 'classification',
    sliding_window_size = 120,
    retrain_every = 50,
    scaling=False
)
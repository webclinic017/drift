#%% Import all the stuff, load data, define constants
from typing import Literal
from sklearnex import patch_sklearn
patch_sklearn()

from load_data import create_target_cum_forward_returns, load_files, create_target_classes
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

from utils.sliding_window import sliding_window_and_flatten

def walk_forward_train_test(
        model_name: str,
        create_model,
        X: pd.DataFrame,
        y: pd.Series,
        window_size: int,
        retrain_every: int
    ):
    print("Training: ", model_name)
    predictions = [None] * (len(y)-1)
    models = [None] * len(predictions)

    train_from = window_size+1
    train_till = len(y)-2
    
    iterations_since_retrain = 0

    for i in range(train_from, train_till):
        # if i % 20 == 0: print('Fold: ', i)
        iterations_since_retrain += 1
        window_start = i - window_size
        window_end = i
        X_train_slice = X[window_start:window_end]
        y_train_slice = y[window_start:window_end]

        if iterations_since_retrain >= retrain_every or models[i-1] is None:
            model = create_model()
            model.fit(X_train_slice, y_train_slice)
            iterations_since_retrain = 0
        else:
            model = models[i-1]
        models[window_end] = model

        predictions[window_end+1] = model.predict(X[window_end+1].reshape(1, -1)).item()
    return models, predictions



regression_models = [
    ('LR', lambda: LinearRegression(n_jobs=-1)),
    ('BayesianRidge', lambda: BayesianRidge()),
    ('KNN', lambda: KNeighborsRegressor(n_neighbors=15)),
    ('MLP', lambda: MLPRegressor(hidden_layer_sizes=(100,20), max_iter=1000)),
    ('AB', lambda: AdaBoostRegressor()),
    # ('RF', lambda: RandomForestRegressor(n_jobs=-1)),
    ('SVR', lambda: SVR(kernel='rbf', C=1e3, gamma=0.1))
]
classification_models = [
    ('LR', lambda: LogisticRegression(n_jobs=-1)),
    ('LDA', lambda: LinearDiscriminantAnalysis()),
    ('KNN', lambda: KNeighborsClassifier()),
    ('CART', lambda: DecisionTreeClassifier()),
    ('NB', lambda: GaussianNB()),
    ('AB', lambda: AdaBoostClassifier()),
    ('RF', lambda: RandomForestClassifier(n_jobs=-1))
]



def run_whole_pipeline(
    ticker_to_predict: str,
    models,
    method: Literal['regression', 'classification'],
    sliding_window_size: int,
    retrain_every: int,
    ):
    print('Predicting: ', ticker_to_predict)

    data = load_files(path='data/',
        own_asset=ticker_to_predict,
        own_asset_lags=[1,2,3,4,5,6,8,10,15],
        load_other_assets=False,
        other_asset_lags=[],
        log_returns=True,
        add_date_features=True,
        own_technical_features='level2',
        other_technical_features='none',
        exogenous_features='none',
        index_column='int'
    )

    target_col = 'target'
    returns_col = ticker_to_predict + '_returns'
    if method == 'regression':
        data = create_target_cum_forward_returns(data, returns_col, 1)
    elif method == 'classification':
        data = create_target_classes(data, returns_col, 1, 'two')
        
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # TODO: should move scaling to an expanding window compomenent
    feature_scaler = MinMaxScaler(feature_range= (-1, 1))
    X = feature_scaler.fit_transform(X)
    # TODO: should scale y as well probably

    X = sliding_window_and_flatten(X, sliding_window_size)
    y = y[sliding_window_size-1:]



    for model_name, create_model in models:

        model_over_time, preds = walk_forward_train_test(
            model_name=model_name,
            create_model = create_model,
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
    retrain_every = 50
)
run_whole_pipeline(
    ticker_to_predict = ticker_to_predict,
    models = classification_models,
    method = 'classification',
    sliding_window_size = 120,
    retrain_every = 50
)
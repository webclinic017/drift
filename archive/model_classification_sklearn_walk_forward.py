#%% Import all the stuff, load data, define constants
from sklearnex import patch_sklearn
patch_sklearn()

from load_data import create_target_classes, load_files
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# from utils.evaluate import print_classification_metrics, format_data_for_backtest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from utils.sliding_window import sliding_window_and_flatten

ticket_to_predict = 'BTC_ETH'
print('Predicting: ', ticket_to_predict)

data = load_files(path='data/',
    own_asset=ticket_to_predict,
    own_asset_lags=[1,2,3,4,5,6,8,10,15],
    load_other_assets=False,
    other_asset_lags=[1,2,3,4],
    log_returns=True,
    add_date_features=True,
    own_technical_features='level1',
    other_technical_features='none',
    exogenous_features='none',
    index_column='int'
)

target_col = 'target'
returns_col = ticket_to_predict + '_returns'
data = create_target_classes(data, returns_col, 1, 'two')
    
X = data.drop(columns=[target_col])
y = data[target_col]

X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_size=0.1)
feature_scaler = MinMaxScaler(feature_range= (-1, 1))
X_test_orig = X_test.copy()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)


#%%

sliding_window_size = 120
retrain_every = 60
X_train = sliding_window_and_flatten(X_train, sliding_window_size)
# X_test = sliding_window_and_flatten(X_test, sliding_window_size)
X_test_orig = X_test_orig.iloc[sliding_window_size-1:]
y_train = y_train[sliding_window_size-1:]
# y_test = y_test[sliding_window_size-1:]




def evaluate_predictions(model_name: str, y: pd.Series, preds: pd.Series, sliding_window_size: int):
    print("Model: ", model_name)
    evaluate_from = sliding_window_size*2
    print(accuracy_score(y[evaluate_from:-1], preds[evaluate_from:]))
    print(confusion_matrix(y[evaluate_from:-1], preds[evaluate_from:]))
    print(classification_report(y[evaluate_from:-1], preds[evaluate_from:]))


def walk_forward_train_test(
        create_model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        window_size: int,
        retrain_every: int
    ):
    predictions = [None] * (len(y_train)-1)
    models = [None] * len(predictions)

    train_from = sliding_window_size+1
    train_till = len(y_train)-2
    
    iterations_since_retrain = 0

    for i in range(train_from, train_till):
        # if i % 20 == 0: print('Fold: ', i)
        iterations_since_retrain += 1
        window_start = i - window_size
        window_end = i
        X_train_slice = X_train[window_start:window_end]
        y_train_slice = y_train[window_start:window_end]

        if iterations_since_retrain >= retrain_every or models[i-1] is None:
            model = create_model()
            model.fit(X_train_slice, y_train_slice)
            iterations_since_retrain = 0
        else:
            model = models[i-1]
        models[window_end] = model

        predictions[window_end+1] = model.predict(X_train[window_end+1].reshape(1, -1)).item()
    return models, predictions


#%%
for model_name, create_model in models_to_try:

    model_over_time, preds = walk_forward_train_test(
        create_model = create_model,
        X_train = X_train,
        y_train = y_train,
        window_size = sliding_window_size,
        retrain_every = retrain_every
    )

    evaluate_predictions(model_name, y_train, preds, sliding_window_size)


#%%


#%% Create column for Strategy Returns by multiplying the daily returns by the position that was held at close of business the previous day
# backtestdata = pd.DataFrame(index= X_test_orig.index)
# backtestdata['signal_pred'] = predictions
# backtestdata['signal_actual'] = y_test
# backtestdata['returns'] = X_test_orig[returns_col]
# backtestdata['only_positive_returns'] = backtestdata['returns'] * backtestdata['signal_actual'].shift(1)
# backtestdata['strategy_returns'] = backtestdata['returns'] * backtestdata['signal_pred'].shift(1)

# %%
# print(backtestdata.cumsum().apply(np.exp).tail(1))

# %%

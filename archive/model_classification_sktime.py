#%% Import all the stuff, load data, define constants
from utils.load_data import load_files, create_target_classes
import pandas as pd
import numpy as np
from utils.sktime import from_df_to_sktime_data
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sktime.classification.interval_based import (
    TimeSeriesForestClassifier,
    SupervisedTimeSeriesForest,

)
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV

from sktime.forecasting.compose import make_reduction

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.model_selection import (
    SlidingWindowSplitter,
    ForecastingGridSearchCV,
)
from utils.evaluate import print_classification_metrics, format_data_for_backtest
from sklearn.ensemble import RandomForestRegressor
from sklearnex import patch_sklearn
patch_sklearn()


ticket_to_predict = 'BTC_USD'
print('Predicting: ', ticket_to_predict)

data = load_files(path='data/',
    own_asset=ticket_to_predict,
    own_asset_lags=[1,2,3,4,5,6,8,10,15],
    load_other_assets=False,
    other_asset_lags=[1,2,3,4],
    log_returns=True,
    add_date_features=True,
    own_technical_features='level2',
    other_technical_features='none',
    exogenous_features='none',
    index_column='int'
)

target_col = 'target'
returns_col = ticket_to_predict + '_returns'
data = create_target_classes(data, returns_col, 1, 'two')

X = data.drop(columns=[target_col])
y = data[target_col]

X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_size=0.2)
X_train = from_df_to_sktime_data(X_train)
X_test = from_df_to_sktime_data(X_test)

#%%

# pipe = RecursiveTabularRegressionForecaster(steps=[
#     # ("deseasonalizer", OptionalPassthrough(Deseasonalizer())),
#     ("scaler", StandardScaler()),
#     ("classifier", TimeSeriesForestClassifier(n_estimators=200, random_state=1)),
# ])

# pipe.fit(X_train, y_train)
# regressor = RandomForestRegressor(n_estimators=20)
# model = DecisionTreeClassifier(random_state=1)
model = TimeSeriesForestClassifier(n_estimators=50, random_state=1)
model.fit(y = y_train, X = X_train)
# forecaster = make_reduction(model, scitype="tabular-regressor")
# nested_params = {"window_length": list(range(2,30)), 
#                  "estimator__max_depth": list(range(5,16))}
#                 # "estimator__n_estimators": list(range(10,200))}
#%%

# cv = SlidingWindowSplitter(initial_window=40, window_length=30)
# nrcv = ForecastingRandomizedSearchCV(forecaster, strategy="refit", cv=cv, 
#                                      param_distributions=nested_params, 
#                                      n_iter=5, random_state=42)
# nrcv.fit(y = y_train, X = X_train, fh=np.array([1]))
# print(nrcv.best_params_)
# print(nrcv.best_score_)

# model = DecisionTreeClassifier(random_state=1)
# model.fit(X_train, y_train)

# preds = nrcv.best_forecaster_.predict( X=X_test)
# # print(preds)
preds = model.predict(X_test)
print(print_classification_metrics(y_test, preds))

# backtest_data = format_data_for_backtest(data, returns_col, X_test, preds)
# print(backtest_data)
# %%

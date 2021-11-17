#%% Import all the stuff, load data, define constants
from load_data import create_target_cum_forward_returns, create_target_pos_neg_classes, load_files, create_target_four_classes
import pandas as pd
import numpy as np
from utils.sktime import from_df_to_sktime_data
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sktime.classification.interval_based import (
    TimeSeriesForestClassifier,
)
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.model_selection import (
    SlidingWindowSplitter,
    ForecastingGridSearchCV,
)
# from sktime.forecasting.model_evaluation import 
# from sklearnex import patch_sklearn
# patch_sklearn()



data = load_files('data/', add_features=True, log_returns=True, narrow_format=False)
data.reset_index(drop=True, inplace=True)
data = data[[column for column in data.columns if not column.endswith('volume')]]
data = data[[column for column in data.columns if column.startswith('BTC_ETH_')]]
target_col = 'target'
data = create_target_pos_neg_classes(data, 'BTC_ETH_returns', 1)

X = data
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

model = TimeSeriesForestClassifier(n_estimators=200, random_state=1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(model.score(X_test, y_test))
print(confusion_matrix(y_test, preds))


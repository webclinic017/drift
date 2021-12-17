#%% Import all the stuff, load data, define constants
from sklearnex import patch_sklearn
patch_sklearn()

from utils.load_data import create_target_cum_forward_returns, create_target_classes, load_files
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
    own_technical_features='level2',
    other_technical_features='none',
    exogenous_features='none',
    index_column='int'
)

target_col = 'target'
returns_col = ticket_to_predict + '_returns'
data = create_target_classes(data, returns_col, 1, 'three')

X = data.drop(columns=['target'])
y = data[target_col]

X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_size=0.2)
feature_scaler = MinMaxScaler(feature_range= (-1, 1))
X_test_orig = X_test.copy()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)


#%%

sliding_window_size = 10
X_train = sliding_window_and_flatten(X_train, sliding_window_size)
X_test = sliding_window_and_flatten(X_test, sliding_window_size)
X_test_orig = X_test_orig.iloc[sliding_window_size-1:]
y_train = y_train[sliding_window_size-1:]
y_test = y_test[sliding_window_size-1:]

assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]
scoring = 'accuracy'

# %%
num_folds = 10

models = []
models.append(('LR', LogisticRegression(n_jobs=-1)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
# models.append(('NN', MLPClassifier(hidden_layer_sizes=[200, 100, 50], shuffle=False)))
models.append(('AB', AdaBoostClassifier()))
# models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier(n_jobs=-1)))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, shuffle=False)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# # compare algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# fig.set_size_inches(15,8)
# plt.show()

#%%
# n_estimators = [20,80]
# max_depth= [5,10, 15]
# criterion = ["gini","entropy"]
# param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, criterion = criterion )
# model = RandomForestClassifier(n_jobs=-1)
# kfold = KFold(n_splits=10, shuffle=False)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
# grid_result = grid.fit(X_train, y_train)

# #Print Results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# ranks = grid_result.cv_results_['rank_test_score']
# for mean, stdev, param, rank in zip(means, stds, params, ranks):
#     print("#%d %f (%f) with: %r" % (rank, mean, stdev, param))


#%% prepare model
model = RandomForestClassifier(criterion='entropy', n_estimators=80, max_depth=5, n_jobs=-1) 
# model = LogisticRegression() 
# model = MLPClassifier(hidden_layer_sizes=[200, 100, 50], shuffle=False, max_iter=1000)
model = GaussianNB()
model.fit(X_train, y_train)


#%%
# estimate accuracy on validation set
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


#%%
# feat_importance = pd.DataFrame({'Importance':model.feature_importances_*100}, index=X.columns)
# feat_importance.sort_values('Importance', axis=0, ascending=True)
# feat_importance.plot(kind='barh', color='r' )
# plt.xlabel('Variable Importance')
# print(feat_importance)

#%% Create column for Strategy Returns by multiplying the daily returns by the position that was held at close of business the previous day
backtestdata = pd.DataFrame(index= X_test_orig.index)
backtestdata['signal_pred'] = predictions
backtestdata['signal_actual'] = y_test
backtestdata['returns'] = X_test_orig[returns_col]
backtestdata['only_positive_returns'] = backtestdata['returns'] * backtestdata['signal_actual'].shift(1)
backtestdata['strategy_returns'] = backtestdata['returns'] * backtestdata['signal_pred'].shift(1)

# %%
print(backtestdata.cumsum().apply(np.exp).tail(1))

# %%

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd

def print_regression_metrics(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    print("RMSE: %.2f" % rmse)

    mae = mean_absolute_error(y_true, y_pred)
    print("MAE: %.2f" % mae)

def print_classification_metrics(y_true, y_pred):
    print("Accuracy: %.2f" % accuracy_score(y_true, y_pred))
    print("Confusion Matrix: \n", confusion_matrix(y_true, y_pred))

def format_data_for_backtest(aggregated_data: pd.DataFrame, returns_col: str, only_test_data: pd.DataFrame, preds) -> pd.DataFrame:
    backtest_data = aggregated_data.iloc[-only_test_data.shape[0]:].copy()[returns_col]
    assert backtest_data.shape[0] == only_test_data.shape[0]
    backtest_data = backtest_data.reset_index(drop=True)    
    return pd.concat([backtest_data, pd.Series(preds)], axis='columns')


# def backtest()
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def print_metrics(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    print("RMSE: %.2f" % rmse)

    mae = mean_absolute_error(y_true, y_pred)
    print("MAE: %.2f" % mae)
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.metrics import confusion_matrix
def print_regression_metrics(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    print("RMSE: %.2f" % rmse)

    mae = mean_absolute_error(y_true, y_pred)
    print("MAE: %.2f" % mae)

def print_classification_metrics(y_true, y_pred):
    print("Accuracy: %.2f" % accuracy_score(y_true, y_pred))
    print("Confusion Matrix: \n", confusion_matrix(y_true, y_pred))
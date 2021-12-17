import numpy as np
import pandas as pd
from training.walk_forward import walk_forward_train_test
from sklearn.base import BaseEstimator

no_of_rows = 100

def __generate_test_data(no_of_rows):
    no_columns = 6
    X = [[row] * no_columns for row in range(no_of_rows)]
    assert X[0][0] == 0
    assert X[1][0] == 1
    assert X[2][0] == 2
    assert X[3][0] == 3
    X = pd.DataFrame(X)

    y = [row+1 for row in range(no_of_rows)]
    assert y[0] == 1
    assert y[1] == 2
    assert y[2] == 3
    y = pd.Series(y)

    return X, y


def test_walk_forward_train_test():
    X, y = __generate_test_data(no_of_rows)

    window_length = 10
    class StubModel(BaseEstimator):

        def fit(self, X, y):
            assert len(X) == window_length
            for i in range(len(X)):
                assert X[i][0] + 1 == y[i]

        def predict(self, X):
            return np.array([X[0][0] + 1])

    model = StubModel()
    scaler = None
    models, predictions = walk_forward_train_test(
        model_name='test',
        model=model,
        X=X,
        y=y,
        target_returns=y,
        window_size=window_length,
        retrain_every=10,
        scaler=scaler)
    for i in range(window_length, no_of_rows):
        predictions[i] == y[i]
    

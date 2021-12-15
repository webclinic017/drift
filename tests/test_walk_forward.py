import pytest
import numpy as np
import pandas as pd
from utils.walk_forward import walk_forward_train_test
from sklearn.base import BaseEstimator

def __generate_test_data():
    no_columns = 6
    no_rows = 100
    X = [[row] * no_columns for row in range(no_rows)]
    assert X[0][0] == 0
    assert X[1][0] == 1
    assert X[2][0] == 2
    assert X[3][0] == 3
    X = pd.DataFrame(X)

    y = [row+1 for row in range(no_rows)]
    assert y[0] == 1
    assert y[1] == 2
    assert y[2] == 3
    y = pd.Series(y)

    return X, y


def test_walk_forward_train_test():
    X, y = __generate_test_data()

    window_length = 10
    class StubModel(BaseEstimator):

        def fit(self, X, y):
            assert len(X) == window_length
            for i in range(len(X)):
                assert X[i][0] + 1 == y[i]

        def predict(self, X):
            return np.array([X[0][0] + 1])

    model = StubModel()
    walk_forward_train_test('test', model, X, y, window_length, 10)

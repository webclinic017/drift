import numpy as np
import pandas as pd
from training.walk_forward import walk_forward_train, walk_forward_inference_batched
from models.base import Model
from sklearn.base import BaseEstimator, ClassifierMixin

no_of_rows = 100


def __generate_incremental_test_data(no_of_rows) -> tuple[pd.DataFrame, pd.Series]:
    """Test data, where X[n][any_column] == y[n]+1"""

    no_columns = 6
    X = [[row] * no_columns for row in range(no_of_rows)]
    assert X[1][0] == 1
    assert X[2][0] == 2
    assert X[3][0] == 3
    assert X[4][0] == 4
    X = pd.DataFrame(X)

    y = [row + 1 for row in range(no_of_rows)]
    assert y[1] == 2
    assert y[2] == 3
    assert y[3] == 4
    y = pd.Series(y)

    return X, y


class IncrementingStubModel(Model, BaseEstimator, ClassifierMixin):
    """
    A deteministic model that can predict the future with 100% accuracy
    It verifies that the X[n][any_column]+1 == y[n]
    """

    data_transformation = "original"
    only_column = None
    predict_window_size = "single_timestamp"

    def __init__(self, window_length) -> None:
        super().__init__()
        self.window_length = window_length

    def fit(self, X, y):
        for i in range(len(X)):
            assert X[i][-1] + 1 == y[i]

    def predict(self, X):
        return np.array([row[0] + 1 for row in X])

    def predict_proba(self, X):
        return np.array([[row[0] + 1, 0] for row in X])


def test_walk_forward_train_test():
    X, y = __generate_incremental_test_data(no_of_rows)

    window_length = 10
    retrain_every = 10

    model = IncrementingStubModel(window_length=window_length)

    model_over_time = walk_forward_train(
        model=model,
        X=X,
        y=y,
        forward_returns=y,
        window_size=window_length,
        retrain_every=retrain_every,
        from_index=None,
        transformations_over_time=[],
    )
    predictions, _ = walk_forward_inference_batched(
        model_name="test",
        model_over_time=model_over_time,
        transformations_over_time=[],
        X=X,
        expanding_window=False,
        window_size=window_length,
        retrain_every=retrain_every,
        class_labels=[0, 1],
        from_index=None,
    )

    # verify if predictions are the same as y
    for i in range(window_length + 2, no_of_rows):
        assert predictions[i] == y[i]

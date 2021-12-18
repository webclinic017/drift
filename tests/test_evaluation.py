
import numpy as np
import pandas as pd
from training.walk_forward import walk_forward_train_test
from sklearn.base import BaseEstimator
from utils.evaluate import evaluate_predictions

no_of_rows = 100

def __generate_even_odd_test_data(no_of_rows) -> tuple[pd.DataFrame, pd.Series]:
    ''' Test data, where X[n][any_column] == 1 if n is even, else 0
    '''
    
    no_columns = 6
    X = [[-1 if row % 2 == 0 else 1] * no_columns for row in range(no_of_rows)]
    assert X[0][0] == -1
    assert X[1][0] == 1
    assert X[2][0] == -1
    assert X[3][0] == 1
    X = pd.DataFrame(X)

    y = [-1 if (row+1) % 2 == 0 else 1 for row in range(no_of_rows)]
    assert y[0] == 1
    assert y[1] == -1
    assert y[2] == 1
    assert y[3] == -1

    y = pd.Series(y)

    return X, y

class EvenOddStubModel(BaseEstimator):
    '''
    A deteministic model that can predict the future with 100% accuracy
    It verifies that the X[n][any_column] == 1 if n is even,
    '''

    def __init__(self, window_length) -> None:
        super().__init__()
        self.window_length = window_length

    def fit(self, X, y):
        assert len(X) == self.window_length
        for i in range(len(X)):
            assert y[i] == -1 if X[i][0] == 1 else 1

    def predict(self, X):
        return np.array([-1 if X[0][0] == 1 else 1])


def test_evaluation():
    X, y = __generate_even_odd_test_data(no_of_rows)
    
    window_length = 10

    model = EvenOddStubModel(window_length = window_length)
    scaler = None
    
    models, predictions = walk_forward_train_test(
        model_name='test',
        model=model,
        X=X,
        y=y,
        target_returns=y,
        window_size=window_length,
        retrain_every=10,
        scaler=scaler
    )
    
    # verify if predictions are the same as y
    for i in range(window_length+2, no_of_rows):
        assert predictions[i] == y[i]

    fake_target_returns = y * 0.1
    processed_predictions_to_match_returns = predictions * 0.1

    result = evaluate_predictions(
        model_name='test',
        target_returns=fake_target_returns,
        y_pred=processed_predictions_to_match_returns,
        method='classification'
    )

    assert result['accuracy'] == 100.0
    

    
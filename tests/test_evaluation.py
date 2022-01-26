
import numpy as np
import pandas as pd
from training.walk_forward import walk_forward_train, walk_forward_inference
from models.base import Model
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

class EvenOddStubModel(Model):
    '''
    A deteministic model that can predict the future with 100% accuracy
    It verifies that the X[n][any_column] == 1 if n is even,
    '''

    data_transformation = "original"
    only_column = None
    predict_window_size = 'single_timestamp'

    def __init__(self, window_length) -> None:
        super().__init__()
        self.window_length = window_length

    def fit(self, X, y):
        assert len(X) == self.window_length
        for i in range(len(X)):
            assert y[i] == -1 if X[i][0] == 1 else 1

    def predict(self, X):
        return (-1 if X[0][0] == 1 else 1, np.array([]))

    def clone(self):
        return self
    
    def get_name(self) -> str:
        return 'test'

    def initialize_network(self, input_dim: int, output_dim: int):
        pass


def test_evaluation():
    X, y = __generate_even_odd_test_data(no_of_rows)
    
    window_length = 10

    model = EvenOddStubModel(window_length = window_length)
    
    model_over_time, transformations_over_time = walk_forward_train(
        model_name='test',
        model=model,
        X=X,
        y=y,
        forward_returns=y,
        expanding_window=False,
        window_size=window_length,
        retrain_every=10,
        from_index=None,
        transformations=[],
        preloaded_transformations=None)
    predictions, _ = walk_forward_inference(
        model_name='test',
        model_over_time=model_over_time,
        transformations_over_time=transformations_over_time,
        X=X,
        expanding_window=False,
        window_size=window_length,
        from_index=None,
    )
    
    # verify if predictions are the same as y
    for i in range(window_length+2, no_of_rows):
        assert predictions[i] == y[i]

    fake_forward_returns = y * 0.1
    processed_predictions_to_match_returns = predictions * 0.1

    result = evaluate_predictions(
        model_name='test',
        forward_returns=fake_forward_returns,
        y_pred=processed_predictions_to_match_returns,
        y_true=y,
        no_of_classes='two',
        print_results = False,
        discretize=True
    )

    assert result['accuracy'] == 100.0
    

    

from __future__ import annotations
from models.base import Model
import numpy as np

class StaticAverageModel(Model):
    '''
    Model that averages .
    '''

    data_scaling = 'unscaled'
    only_column = 'model_'
    feature_selection = 'off'
    model_type = 'static'
    predict_window_size = 'single_timestamp'

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # This is a static model, it can' learn anything
        pass

    def predict(self, X) -> tuple[float, np.ndarray]:
        # Make sure there's data to average
        assert X.shape[1] > 0
        prediction = np.average(X[-1])
        return (prediction, np.array([]))

    def clone(self) -> StaticAverageModel:
        return self

    def get_name(self) -> str:
        return 'static_average'
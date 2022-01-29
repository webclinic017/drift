from __future__ import annotations
from models.base import Model
import numpy as np

class StaticMomentumModel(Model):
    '''
    Model that uses only one feature: momentum. It's positive if momentum is greater than 0, otherwise it's negative.
    '''

    method = 'classification'
    data_transformation = 'original'
    only_column = 'mom'
    model_type = 'static'
    predict_window_size = 'single_timestamp'

    def __init__(self, allow_short: bool) -> None:
        super().__init__()
        self.allow_short = allow_short

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # This is a static model, it can' learn anything
        pass

    def predict(self, X) -> tuple[float, np.ndarray]:
        negative_class = -1.0 if self.allow_short == True else 0.0
        prediction = 1.0 if X[-1][0] > 0 else negative_class
        return (prediction, np.array([]))

    def clone(self) -> StaticMomentumModel:
        return self

    def initialize_network(self, input_dim:int, output_dim:int):
        pass
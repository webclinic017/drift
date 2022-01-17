from __future__ import annotations
from models.base import Model
import numpy as np

class StaticNaiveModel(Model):
    '''
    Model that carries the last observation (from returns) to the next one, naively.
    '''

    method = 'regression'
    data_transformation = 'original'
    only_column = None
    model_type = 'static'
    predict_window_size = 'single_timestamp'

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # This is a static model, it can' learn anything
        pass

    def predict(self, X) -> tuple[float, np.ndarray]:
        return (X[-1][0], np.array([]))

    def clone(self) -> StaticNaiveModel:
        return self

    def get_name(self) -> str:
        return 'static_naive'

    def initialize_network(self, input_dim:int, output_dim:int):
        pass
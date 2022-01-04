from __future__ import annotations
from models.base import Model
import numpy as np

class StaticMomentumModel(Model):
    '''
    Model that uses only one feature: momentum. It's positive if momentum is greater than 0, otherwise it's negative.
    '''

    data_scaling = 'unscaled'
    only_column = 'mom'
    feature_selection = 'off'
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
    
    def get_name(self) -> str:
        return 'static_mom'
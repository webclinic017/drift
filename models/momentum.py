from __future__ import annotations
import numpy as np
from .base import Model
from sklearn.base import BaseEstimator, ClassifierMixin

class StaticMomentumModel(BaseEstimator, ClassifierMixin, Model):
    '''
    Model that uses only one feature: momentum. It's positive if momentum is greater than 0, otherwise it's negative.
    '''

    data_transformation = 'original'
    only_column = 'mom'
    predict_window_size = 'single_timestamp'

    def __init__(self, allow_short: bool) -> None:
        super().__init__()
        self.allow_short = allow_short

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # This is a static model, it can' learn anything
        pass

    def predict(self, X) -> np.ndarray:
        negative_class = -1.0 if self.allow_short == True else 0.0
        prediction = 1.0 if X[-1][0] > 0 else negative_class
        return np.array(prediction)
    
    def predict_proba(self, X) -> np.ndarray:
        return np.array([])

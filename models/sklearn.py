from __future__ import annotations
from models.base import Model
import numpy as np
from sklearn.base import clone


class SKLearnModel(Model):

    data_scaling = 'scaled'
    only_column = None
    feature_selection = 'on'
    model_type = 'ml'
    predict_window_size = 'single_timestamp'

    def __init__(self, model):
        self.model = model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X) -> tuple[float, np.ndarray]:
        pred = self.model.predict(X).item()
        probability = self.model.predict_proba(X).squeeze()
        return (pred, probability)
    
    def clone(self) -> SKLearnModel:
        return SKLearnModel(clone(self.model))

    def get_name(self) -> str:
        return self.model.__class__.__name__
    
    def initialize_network(self, input_dim:int, output_dim:int):
        pass
    
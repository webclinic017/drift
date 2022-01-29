from __future__ import annotations
from typing import Literal
from models.base import Model
import numpy as np
from sklearn.base import clone


class SKLearnModel(Model):

    method: Literal["regression", "classification"]
    data_transformation = 'transformed'
    only_column = None
    model_type = 'ml'
    predict_window_size = 'single_timestamp'

    def __init__(self, model, method: Literal['regression', 'classification']):
        self.model = model
        self.method = method
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X) -> tuple[float, np.ndarray]:
        pred = self.model.predict(X).item()
        probability = self.model.predict_proba(X).squeeze()
        return (pred, probability)
    
    def clone(self) -> SKLearnModel:
        return SKLearnModel(clone(self.model), self.method)
    
    def initialize_network(self, input_dim:int, output_dim:int):
        pass
    
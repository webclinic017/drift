from __future__ import annotations
from models.base import Model
import numpy as np
from xgboost import XGBClassifier
from sklearn.base import clone

class XGBoostModel(Model):

    method = 'classification'
    data_transformation = 'transformed'
    only_column = None
    model_type = 'ml'
    predict_window_size = 'single_timestamp'

    def __init__(self, model: XGBClassifier):
        self.model = model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        def map_to_xgb(y): return np.array([1 if i == 1 else 0 for i in y])
        self.model.fit(X, map_to_xgb(y))

    def predict(self, X) -> tuple[float, np.ndarray]:
        pred = self.model.predict(X).item()
        probability = self.model.predict_proba(X).squeeze()
        def map_from_xgb(y): return 1 if y == 1 else -1
        return (map_from_xgb(pred), probability)
    
    def clone(self) -> XGBoostModel:
        return XGBoostModel(clone(self.model))

    def get_name(self) -> str:
        return self.model.__class__.__name__
    
    def initialize_network(self, input_dim:int, output_dim:int):
        pass
    
from __future__ import annotations
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from models.base import Model
import numpy as np
from copy import deepcopy


class StatsModel(Model):

    # This is work in progress
    data_transformation = 'transformed'
    only_column = None
    model_type = 'ml'
    predict_window_size = 'single_timestamp'

    def __init__(self, model: TimeSeriesModel):
        self.model = model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X) -> tuple[float, np.ndarray]:
        pred = self.model.predict(X).item()
        return (pred, np.array([0]))
    
    def clone(self) -> StatsModel:
        return StatsModel(deepcopy(self.model))
    
    def initialize_network(self, input_dim:int, output_dim:int):
        pass
    
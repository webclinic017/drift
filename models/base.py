from __future__ import annotations
from typing import Literal, Optional
from sklearn.base import clone
from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):

    data_scaling: Literal["scaled", "unscaled"]
    feature_selection: Literal["on", "off"]
    # data_format: Literal["wide", "narrow"]
    only_column: Optional[str]
    model_type: Literal['ml', 'static']
    predict_window_size: Literal['single_timestamp', 'window_size']

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X) -> tuple[float, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> Model:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError


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
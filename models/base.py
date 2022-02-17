from __future__ import annotations
from typing import Literal, Optional, Union
from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):

    name: str = ""
    data_transformation: Literal["transformed", "original"]
    only_column: Optional[str]
    predict_window_size: Literal['single_timestamp', 'window_size']

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


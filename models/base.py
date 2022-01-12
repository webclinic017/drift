from __future__ import annotations
from typing import Literal, Optional, Union
from abc import ABC, abstractmethod
import numpy as np

# import numpy as np

class Model(ABC):

    data_transformation: Literal["transformed", "original"]
    feature_selection: Literal["on", "off"]
    # data_format: Literal["wide", "narrow"]
    only_column: Optional[str]
    model_type: Literal['ml', 'static']
    predict_window_size: Literal['single_timestamp', 'window_size']

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[float, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> Model:
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def initialize_network(self, input_dim:int, output_dim:int):
        pass


    
    

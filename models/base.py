from __future__ import annotations
from typing import Literal, Optional, Union
from sklearn.base import clone
from abc import ABC, abstractmethod
import numpy as np

import copy
import pytorch_lightning as pl

import numpy as np
from data_loader.pytorch_dataset import get_dataloader

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
    
    
    
class LightningNeuralNetModel(Model):

    data_scaling = 'scaled'
    only_column = None
    feature_selection = 'off'
    model_type = 'ml'

    ''' Standard lightning methods '''
    
    def __init__(self, model, max_epochs=5):
        self.model = model
        self.trainer = pl.Trainer(max_epochs=max_epochs)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        train_dataloader = self.__prepare_data(X.astype(float), y.astype(float))
        self.trainer.fit(self.model, train_dataloader)
        
    def predict(self, X: np.ndarray) -> tuple[float, np.ndarray]:
        return self.model(X)

    def clone(self):
        model_copy = copy.deepcopy(self.model)
        return LightningNeuralNetModel(model_copy)
    
    ''' Non-standard lightning methods '''
    def __prepare_data(self, X:np.ndarray, y:np.ndarray):
        dataloader = get_dataloader(X, y)
        return dataloader

    def initialize_network(self, input_dim:int, output_dim:int):
        self.model.initialize_network(input_dim, output_dim)

    def get_name(self) -> str:
        return self.model.__class__.__name__
    
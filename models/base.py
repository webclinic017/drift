
from typing import Literal, Optional
from sklearn.base import clone
from abc import ABC, abstractmethod, abstractproperty

class Model(ABC):

    data_scaling: Literal["scaled", "unscaled"]
    feature_selection: Literal["on", "off"]
    # data_format: Literal["wide", "narrow"]
    only_column: Optional[str]
    model_type: Literal['ml', 'static']

    @abstractmethod
    def fit(self, X, y, prev_model):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def clone(self):
        pass


class SKLearnModel(Model):

    data_scaling = 'scaled'
    only_column = None
    feature_selection = 'on'
    model_type = 'ml'

    def __init__(self, model):
        self.model = model

    def fit(self, X, y, prev_model):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def clone(self):
        return SKLearnModel(clone(self.model))
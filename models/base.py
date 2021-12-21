
from typing import Literal, Optional
from sklearn.base import clone
from abc import ABC, abstractmethod, abstractproperty

class Model(ABC):

    # data_format: Literal['dataframe', 'numpy'] 
    data_scaling: Literal["scaled", "unscaled"]
    # data_format: Literal["wide", "narrow"]
    only_column: Optional[str]

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def clone(self):
        pass


class SKLearnModel(Model):

    # data_format = 'numpy'
    data_scaling = 'scaled'
    only_column = None

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def clone(self):
        return SKLearnModel(clone(self.model))
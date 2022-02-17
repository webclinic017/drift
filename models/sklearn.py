from __future__ import annotations
from typing import Literal
from .base import Model
import numpy as np


def SKLearnModel(instance) -> Model:

    instance.data_transformation = "transformed"
    instance.only_column = None
    instance.predict_window_size = "single_timestamp"
    instance.name = instance.__class__.__name__

    return instance

    # def predict(self, X) -> tuple[float, np.ndarray]:
    #     pred = self.model.predict(X).item()
    #     probability = self.model.predict_proba(X).squeeze()
    #     return (pred, probability)

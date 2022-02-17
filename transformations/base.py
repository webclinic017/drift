from __future__ import annotations
from typing import Literal, Optional, Union
from abc import ABC, abstractmethod
import pandas as pd


class Transformation(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> Transformation:
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

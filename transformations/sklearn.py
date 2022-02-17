from __future__ import annotations
from transformations.base import Transformation
from typing import Literal, Optional, Union
from sklearn.base import clone, BaseEstimator
import pandas as pd


class SKLearnTransformation(Transformation):

    transformer: BaseEstimator

    def __init__(self, transformer: BaseEstimator):
        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.transformer.fit(X, y)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series]) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.transformer.transform(X), index=X.index, columns=X.columns
        )

    def clone(self) -> SKLearnTransformation:
        return SKLearnTransformation(clone(self.transformer))

    def get_name(self) -> str:
        return self.transformer.__class__.__name__

from __future__ import annotations
from transformations.base import Transformation
from typing import Optional
from copy import deepcopy
from sklearn.feature_selection import RFE
import pandas as pd
from models.sklearn import SKLearnModel


class RFETransformation(Transformation):

    rfe: RFE
    n_feature_to_select: int

    def __init__(self, n_feature_to_select: int, model: SKLearnModel, step=0.1):
        self.n_feature_to_keep = n_feature_to_select
        self.model = model
        self.rfe = RFE(model, n_features_to_select=n_feature_to_select, step=step)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        if self.rfe is None:
            return
        self.rfe.fit(X, y)

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        if self.rfe is None:
            return X
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.rfe is None:
            return X
        return pd.DataFrame(X[X.columns[self.rfe.support_]], index=X.index)

    def clone(self) -> RFETransformation:
        return deepcopy(self)

    def get_name(self) -> str:
        return "RFE"

from .rfe import RFETransformation
from .pca import PCATransformation
from .sklearn import SKLearnTransformation
from typing import Literal, Optional
from models.model_map import default_feature_selector_classification
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    StandardScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
)

ScalerTypes = Literal[
    "normalize", "minmax", "standardize", "robust", "box-cox", "quantile"
]


def get_rfe(n_feature_to_select: int) -> Optional[RFETransformation]:
    if n_feature_to_select > 0:
        return RFETransformation(
            n_feature_to_select=n_feature_to_select,
            model=default_feature_selector_classification,
        )
    else:
        return None


def get_pca(
    ratio_components_to_keep: float, initial_window_size: int
) -> Optional[PCATransformation]:

    if ratio_components_to_keep > 0:
        return PCATransformation(
            ratio_components_to_keep=ratio_components_to_keep,
            initial_window_size=initial_window_size,
        )
    else:
        return None


def get_scaler(type: ScalerTypes) -> SKLearnTransformation:
    if type == "normalize":
        return SKLearnTransformation(Normalizer())
    elif type == "minmax":
        return SKLearnTransformation(MinMaxScaler(feature_range=(-1, 1)))
    elif type == "standardize":
        return SKLearnTransformation(StandardScaler())
    elif type == "robust":
        return SKLearnTransformation(
            RobustScaler(with_centering=False, quantile_range=(0.10, 0.90))
        )
    elif type == "box-cox":
        return SKLearnTransformation(
            PowerTransformer(method="box-cox", standardize=True)
        )
    elif type == "quantile":
        return SKLearnTransformation(
            QuantileTransformer(n_quantiles=100, output_distribution="normal")
        )
    else:
        raise Exception("Scaler type not supported")

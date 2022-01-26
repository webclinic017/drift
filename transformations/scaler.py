from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from .sklearn import SKLearnTransformation
from typing import Literal

ScalerTypes = Literal['normalize', 'minmax', 'standardize']

def get_scaler(type: ScalerTypes) -> SKLearnTransformation:
    if type == 'normalize':
        return SKLearnTransformation(Normalizer())
    elif type == 'minmax':
        return SKLearnTransformation(MinMaxScaler(feature_range= (-1, 1)))
    elif type == 'standardize':
        return SKLearnTransformation(StandardScaler())
    else:
        raise Exception("Scaler type not supported")
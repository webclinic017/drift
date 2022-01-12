from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from typing import Union
from utils.types import ScalerTypes

def get_scaler(type: ScalerTypes) -> Union[MinMaxScaler, Normalizer, StandardScaler]:
    if type == 'normalize':
        return Normalizer()
    elif type == 'minmax':
        return MinMaxScaler(feature_range= (-1, 1))
    elif type == 'standardize':
        return StandardScaler()
    else:
        raise Exception("Scaler type not supported")
from sklearn.feature_selection import RFE
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from models.base import Model, SKLearnModel
from utils.scaler import get_scaler
from utils.types import ScalerTypes
from utils.hashing import hash_df, hash_series
from diskcache import Cache
cache = Cache(".cachedir/feature_selection")

def select_features(**kwargs) -> pd.DataFrame:
    hashed = kwargs['data_config_hash'] + kwargs['model'].get_name() + str(kwargs['n_features_to_select']) + kwargs['backup_model'].get_name() + kwargs['scaling']
    if hashed in cache:
        return cache.get(hashed)
    else:
        return_value = __select_features(**kwargs)
        cache[hashed] = return_value
        return return_value

def __select_features(X: pd.DataFrame, y: pd.Series, model: Model, n_features_to_select: int, backup_model: SKLearnModel, scaling: ScalerTypes, data_config_hash: str) -> pd.DataFrame:
    ''' Select features using RFECV, returns a pd.DataFrame (X) with only the selected features.'''
    if model.model_type != 'ml': return X

    # 2. Recursive feature selection
    cv = TimeSeriesSplit(n_splits=5)
    scaler = get_scaler(scaling)
    X_scaled = X.copy()
    if scaler is not None:
        X_scaled = scaler.fit_transform(X_scaled)

    feat_selector_model = model.model
    if hasattr(feat_selector_model, 'feature_importances_') == False and hasattr(feat_selector_model, 'coef_') == False:
        feat_selector_model = backup_model.model

#     selector = RFECV(feat_selector_model, cv = cv, step=5, min_features_to_select=min_features_to_select)
    selector = RFE(feat_selector_model, n_features_to_select= n_features_to_select, step=5)
    selector = selector.fit(X_scaled, y)
    print("Kept %d features out of %d" % (selector.n_features_, X_scaled.shape[1]))

    return pd.DataFrame(X[X.columns[selector.support_]], index= X.index)

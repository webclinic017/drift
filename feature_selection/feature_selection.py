from sklearn.feature_selection import RFE
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from models.base import Model, SKLearnModel
from sklearn.decomposition import PCA

def select_features(X: pd.DataFrame, y: pd.Series, model: Model, n_features_to_select: int, backup_model: SKLearnModel) -> pd.DataFrame:
    ''' Select features using RFECV, returns a pd.DataFrame (X) with only the selected features.'''
    if model.model_type != 'ml': return X

    # 2. Recursive feature selection
    cv = TimeSeriesSplit(n_splits=5)

    feat_selector_model = model.model
    if hasattr(feat_selector_model, 'feature_importances_') == False and hasattr(feat_selector_model, 'coef_') == False:
        feat_selector_model = backup_model.model

#     selector = RFECV(feat_selector_model, cv = cv, step=5, min_features_to_select=min_features_to_select)
    selector = RFE(feat_selector_model, n_features_to_select= n_features_to_select)
    selector = selector.fit(X, y)
    print("Kept %d features out of %d" % (selector.n_features_, X.shape[1]))

    return pd.DataFrame(X[X.columns[selector.support_]], index= X.index)

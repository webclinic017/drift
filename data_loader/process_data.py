import pandas as pd
from operator import itemgetter

from utils.helpers import has_enough_samples_to_train
from feature_selection.dim_reduction import reduce_dimensionality
from models.model_map import default_feature_selector_regression, default_feature_selector_classification
from feature_selection.feature_selection import select_features
import warnings




def process_data(X:pd.DataFrame, y:pd.Series, configs: dict) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    
    model_config, training_config, data_config = itemgetter('model_config', 'training_config', 'data_config')(configs)
    
    original_X = X.copy()

    # 2a. Dimensionality Reduction (optional)
    if training_config['dimensionality_reduction']:
        X_pca = reduce_dimensionality(X, int(len(X.columns) / 2))
        X = X_pca.copy()
    else:
        X_pca = X.copy()
    
    # 2b. Feature Selection
    print("Feature Selection started")
    # TODO: this needs to be done per model!
    backup_model = default_feature_selector_regression if data_config['method'] == 'regression' else default_feature_selector_classification
    X = select_features(X = X, y = y, model = model_config['primary_models'][0][1], n_features_to_select = training_config['n_features_to_select'], backup_model = backup_model, scaling = training_config['scaler'])
    
    return X, original_X, X_pca

def check_data(X:pd.DataFrame, y:pd.Series, training_config:dict):
    """ Returns True if data is valid, else returns False."""
    
    if has_enough_samples_to_train(X, y, training_config) == False:
        warnings.warn("Not enough samples to train")
        return False
    
    return True
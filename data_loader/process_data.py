import pandas as pd

from utils.helpers import has_enough_samples_to_train
import warnings

def check_data(X:pd.DataFrame, y:pd.Series, training_config:dict):
    """ Returns True if data is valid, else returns False."""
    
    if has_enough_samples_to_train(X, y, training_config) == False:
        warnings.warn("Not enough samples to train")
        return False
    
    return True
import pandas as pd
from config.config import Config
from utils.helpers import has_enough_samples_to_train
import warnings

def check_data(X:pd.DataFrame, y:pd.Series, config: Config):
    """ Returns True if data is valid, else returns False."""
    
    if has_enough_samples_to_train(X, y, config) == False:
        warnings.warn("Not enough samples to train")
        return False
    
    return True
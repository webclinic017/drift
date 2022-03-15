import pandas as pd
from config.types import Config
import warnings
from utils.helpers import get_first_valid_return_index
from data_loader.types import XDataFrame


def check_data(X: XDataFrame, config: Config) -> bool:
    """Returns True if data is valid, else returns False."""

    if has_enough_samples_to_train(X, config) == False:
        warnings.warn("Not enough samples to train")
        return False

    return True


def has_enough_samples_to_train(X: XDataFrame, config: Config) -> bool:
    first_valid_index = get_first_valid_return_index(X.iloc[:, 0])
    samples_to_train = len(X) - first_valid_index
    return samples_to_train > config.retrain_every * 3 + config.initial_window_size

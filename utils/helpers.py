import pandas as pd
import numpy as np

def get_first_valid_return_index(series: pd.Series) -> int:
    first_nonzero_return = np.where(np.logical_and(series != 0, np.logical_not(np.isnan(series))))[0][0]
    return first_nonzero_return
import pandas as pd

def normalize(data: pd.DataFrame) -> pd.DataFrame:
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    return (data - data_mean) / data_std

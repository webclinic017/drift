import pandas as pd
from training.types import TransformationsOverTime
from utils.helpers import get_first_valid_return_index
from tqdm import tqdm
from transformations.base import Transformation
from typing import Optional
from data_loader.types import ForwardReturnSeries, XDataFrame, ySeries
import ray

def walk_forward_process_transformations(
                        X: XDataFrame,
                        y: ySeries,
                        forward_returns: ForwardReturnSeries,
                        expanding_window: bool,
                        window_size: int,
                        retrain_every: int,
                        from_index: Optional[pd.Timestamp],
                        transformations: list[Transformation],
                    ) -> TransformationsOverTime:
    transformations_over_time = [pd.Series(index=y.index).rename(t.get_name()) for t in transformations]

    first_nonzero_return = max(get_first_valid_return_index(forward_returns), get_first_valid_return_index(X.iloc[:,0]), get_first_valid_return_index(y))
    train_from = first_nonzero_return + window_size + 1 if from_index is None else X.index.to_list().index(from_index)
    train_till = len(y)
    
    processed_transformations = ray.get([preprocess_transformations_window.remote(X, y, expanding_window, window_size, transformations, first_nonzero_return, index) for index in range(train_from, train_till, retrain_every)])
        
    for transformation, index_time in processed_transformations:
        for transformation_index, transformation in enumerate(transformation):
            transformations_over_time[transformation_index][X.index[index_time]] = transformation

    return transformations_over_time

@ray.remote
def preprocess_transformations_window(X: XDataFrame, y: ySeries, expanding_window: bool, window_size: int, transformations: list[Transformation], first_nonzero_return: int, index: int) -> tuple[list[Transformation], int]:
    train_window_start = X.index[first_nonzero_return] if expanding_window else X.index[index - window_size - 1]
    train_window_end = X.index[index - 1]

    X_expanding_window = X[train_window_start:train_window_end]
    y_expanding_window = y[train_window_start:train_window_end]

    current_transformations = [t.clone() for t in transformations]
    for transformation in current_transformations:
        X_expanding_window = transformation.fit_transform(X_expanding_window, y_expanding_window)

    return (current_transformations, index)
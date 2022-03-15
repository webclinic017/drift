import pandas as pd
from models.base import Model
from training.types import ModelOverTime, TransformationsOverTime
from utils.helpers import get_first_valid_return_index
from tqdm import tqdm
from typing import Optional
from data_loader.types import ForwardReturnSeries, XDataFrame, ySeries
import ray
from utils.parallel import parallel_compute_with_bar
from copy import deepcopy


def walk_forward_train(
    model: Model,
    X: XDataFrame,
    y: ySeries,
    forward_returns: ForwardReturnSeries,
    window_size: int,
    retrain_every: int,
    from_index: Optional[pd.Timestamp],
    transformations_over_time: TransformationsOverTime,
) -> ModelOverTime:
    models_over_time = pd.Series(index=y.index).rename(model.name)

    first_nonzero_return = max(
        get_first_valid_return_index(forward_returns),
        get_first_valid_return_index(X.iloc[:, 0]),
        get_first_valid_return_index(y),
    )
    train_from = (
        first_nonzero_return + window_size + 1
        if from_index is None
        else X.index.to_list().index(from_index)
    )
    train_till = len(y)

    if model.only_column is not None:
        X = X[[column for column in X.columns if model.only_column in column]]

    if model.data_transformation == "original":
        transformations_over_time = []

    models = parallel_compute_with_bar(
        [
            train_on_window.remote(
                index,
                first_nonzero_return,
                X,
                y,
                model,
                transformations_over_time,
            )
            for index in tqdm(range(train_from, train_till, retrain_every))
        ]
    )
    for index, current_model in models:
        models_over_time[X.index[index]] = current_model

    return models_over_time


@ray.remote
def train_on_window(
    index: int,
    first_nonzero_return: int,
    X: XDataFrame,
    y: ySeries,
    model: Model,
    transformations_over_time: TransformationsOverTime,
) -> tuple[int, Model]:
    train_window_start = X.index[first_nonzero_return]

    train_window_end = X.index[index - 1]
    current_transformations = [
        transformation_over_time[index]
        for transformation_over_time in transformations_over_time
    ]
    X_slice = X[train_window_start:train_window_end]

    for transformation in current_transformations:
        X_slice = transformation.transform(X_slice)

    X_slice = X_slice.to_numpy()
    y_slice = y[train_window_start:train_window_end].to_numpy()

    current_model = deepcopy(model)

    current_model.fit(X_slice, y_slice)
    return index, model

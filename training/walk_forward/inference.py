import pandas as pd
from training.types import (
    ModelOverTime,
    TransformationsOverTime,
    PredictionsSeries,
    ProbabilitiesDataFrame,
)
from utils.helpers import get_first_valid_return_index
from tqdm import tqdm
from typing import Optional
from data_loader.types import XDataFrame
from utils.helpers import get_last_non_na_index


def walk_forward_inference(
    model_name: str,
    model_over_time: ModelOverTime,
    transformations_over_time: TransformationsOverTime,
    X: XDataFrame,
    expanding_window: bool,
    window_size: int,
    retrain_every: int,
    class_labels: list[int],
    from_index: Optional[pd.Timestamp],
) -> tuple[PredictionsSeries, ProbabilitiesDataFrame]:
    predictions = pd.Series(index=X.index, dtype="object").rename(model_name)
    probabilities = pd.DataFrame(
        index=X.index, columns=[str(label) for label in class_labels]
    )

    inference_from = (
        max(
            get_first_valid_return_index(model_over_time),
            get_first_valid_return_index(X.iloc[:, 0]),
        )
        if from_index is None
        else X.index.to_list().index(from_index)
    )
    inference_till = X.shape[0]
    model_index_offset = (
        get_last_non_na_index(model_over_time, inference_from)
        if pd.isna(model_over_time[inference_from])
        else 0
    )
    first_model = (
        model_over_time[inference_from - model_index_offset]
        if pd.isna(model_over_time[inference_from])
        else model_over_time[inference_from]
    )

    if first_model.only_column is not None:
        X = X[[column for column in X.columns if first_model.only_column in column]]

    if first_model.data_transformation == "original":
        transformations_over_time = []

    for index in tqdm(range(inference_from, inference_till)):

        last_model_index = (
            index - ((index - inference_from) % retrain_every) - model_index_offset
        )
        train_window_start = (
            X.index[inference_from]
            if expanding_window
            else X.index[index - window_size - 1]
        )

        current_model = model_over_time[X.index[last_model_index]]
        current_transformations = [
            transformation_over_time[X.index[last_model_index]]
            for transformation_over_time in transformations_over_time
        ]

        if current_model.predict_window_size == "window_size":
            next_timestep = X.loc[train_window_start : X.index[index]]
        else:
            # we need to get a Dataframe out of it, since the transformation step always expects a 2D array, but it's equivalent to X.iloc[index]
            next_timestep = X.loc[X.index[index] : X.index[index]]

        for transformation in current_transformations:
            next_timestep = transformation.transform(next_timestep)

        next_timestep = next_timestep.to_numpy()

        prediction = current_model.predict(next_timestep)
        probs = current_model.predict_proba(next_timestep)

        predictions[X.index[index]] = prediction
        if inference_from == index and len(probabilities.columns) != len(probs):
            probabilities = probabilities.reindex(
                columns=["prob_" + str(num) for num in range(0, len(probs.T))]
            )
        probabilities.loc[X.index[index]] = probs

    return predictions, probabilities

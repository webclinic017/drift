import pandas as pd
from models.base import Model
from training.types import ModelOverTime, TransformationsOverTime, PredictionsSeries, ProbabilitiesDataFrame
from transformations.base import Transformation
from utils.helpers import get_first_valid_return_index
from tqdm import tqdm
from typing import Optional
from data_loader.types import XDataFrame
import ray

def walk_forward_inference(
                    model_name: str,
                    model_over_time: ModelOverTime,
                    transformations_over_time: TransformationsOverTime,
                    X: XDataFrame,
                    expanding_window: bool,
                    window_size: int,
                    retrain_every: int,
                    from_index: Optional[pd.Timestamp],
                ) -> tuple[PredictionsSeries, ProbabilitiesDataFrame]:
    predictions = pd.Series(index=X.index, dtype='object').rename(model_name)
    probabilities = pd.DataFrame(index=X.index)

    inference_from = max(get_first_valid_return_index(model_over_time), get_first_valid_return_index(X.iloc[:,0])) if from_index is None else X.index.to_list().index(from_index)
    inference_till = X.shape[0]
    first_model = model_over_time[inference_from]

    if first_model.only_column is not None:
        X = X[[column for column in X.columns if first_model.only_column in column]]
    
    if first_model.data_transformation == 'original':
        transformations_over_time = []

    batch_size = int((inference_till - inference_from) / 10)
    batched_results = ray.get([__inference_from_window.remote(index, index + batch_size, inference_from, retrain_every, X, model_over_time, transformations_over_time, expanding_window, window_size) for index in range(inference_from, inference_till)])
    for batch in batched_results:
        for index, prediction, probs in batch:
            predictions[X.index[index]] = prediction
            probabilities.loc[X.index[index]] = probs

    return predictions, probabilities

@ray.remote
def __inference_from_window(index_start: int, index_end: int, inference_from: int, retrain_every: int, X: XDataFrame, model_over_time: ModelOverTime, transformations_over_time: TransformationsOverTime, expanding_window: bool, window_size: int) -> list[tuple[int, float, pd.Series]]:
    
    results = []
    for index in range(index_start, index_end):
        last_model_index = index - ((index - inference_from) % retrain_every)
        train_window_start = X.index[inference_from] if expanding_window else X.index[index - window_size - 1]

        current_model = model_over_time[X.index[last_model_index]]
        current_transformations = [transformation_over_time[X.index[last_model_index]] for transformation_over_time in transformations_over_time]

        if current_model.predict_window_size == 'window_size': 
            next_timestep = X.loc[train_window_start:X.index[index]]
        else: 
            # we need to get a Dataframe out of it, since the transformation step always expects a 2D array, but it's equivalent to X.iloc[index]
            next_timestep = X.loc[X.index[index]:X.index[index]]
        
        for transformation in current_transformations:
            next_timestep = transformation.transform(next_timestep)

        next_timestep = next_timestep.to_numpy()

        prediction, probs = current_model.predict(next_timestep)
        results.append((index, prediction, probs))
    
    return results
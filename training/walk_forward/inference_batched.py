import pandas as pd
from training.types import ModelOverTime, TransformationsOverTime, PredictionsSeries, ProbabilitiesDataFrame
from utils.helpers import get_first_valid_return_index
from tqdm import tqdm
from typing import Optional
from data_loader.types import XDataFrame
from tqdm import tqdm

def walk_forward_inference_batched(
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
    probabilities = pd.DataFrame(index=X.index, columns=['0', '1'])

    inference_from = max(get_first_valid_return_index(model_over_time), get_first_valid_return_index(X.iloc[:,0])) if from_index is None else X.index.to_list().index(from_index)
    inference_till = X.shape[0]
    first_model = model_over_time[inference_from]

    if first_model.only_column is not None:
        X = X[[column for column in X.columns if first_model.only_column in column]]
    
    if first_model.data_transformation == 'original':
        transformations_over_time = []

    batch_indices = range(inference_from, inference_till, retrain_every) if inference_till - inference_from > retrain_every else [inference_from]
    batched_results = [__inference_from_window(index, index + retrain_every, X, model_over_time, transformations_over_time, expanding_window, window_size) for index in tqdm(batch_indices)]
    for batch in batched_results:
        for index, prediction, probs in batch:
            predictions[X.index[index]] = prediction
            probabilities.loc[X.index[index]] = probs

    return predictions, probabilities

def __inference_from_window(index_start: int, index_end: int, X: XDataFrame, model_over_time: ModelOverTime, transformations_over_time: TransformationsOverTime, expanding_window: bool, window_size: int) -> list[tuple[int, float, pd.Series]]:
    current_model = model_over_time[X.index[index_start]]
    current_transformations = [transformation_over_time[X.index[index_start]] for transformation_over_time in transformations_over_time]

    input_data = X.iloc[index_start:index_end]
        
    for transformation in current_transformations:
        input_data = transformation.transform(input_data)

    input_data = input_data.to_numpy()

    predictions = current_model.predict(input_data)
    probs = current_model.predict_proba(input_data)
    results = [(index_start + index, predictions[index], probs[index]) for index in range(len(predictions))]
    
    return results
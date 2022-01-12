import pandas as pd
from models.base import Model
import numpy as np
from utils.helpers import get_first_valid_return_index
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from typing import Union
from sklearn.base import clone
from transformations.base import Transformation

def walk_forward_train(
                        model_name: str,
                        model: Model,
                        X: pd.DataFrame,
                        y: pd.Series,
                        target_returns: pd.Series,
                        expanding_window: bool,
                        window_size: int,
                        retrain_every: int,
                        transformations: list[Transformation],
                    ) -> tuple[pd.Series, list[pd.Series]]:
    assert len(X) == len(y)
    models_over_time = pd.Series(index=y.index).rename(model_name)
    transformations_over_time = [pd.Series(index=y.index).rename(t.get_name()) for t in transformations]

    first_nonzero_return = max(get_first_valid_return_index(target_returns), get_first_valid_return_index(X.iloc[:,0]), get_first_valid_return_index(y))
    train_from = first_nonzero_return + window_size + 1
    train_till = len(y)
    iterations_before_retrain = 0

    if model.only_column is not None:
        X = X[[column for column in X.columns if model.only_column in column]]
    
    if model.data_transformation == 'original':
        transformations = []
    
    for index in tqdm(range(train_from, train_till)):
        train_window_start = first_nonzero_return if expanding_window else index - window_size - 1

        if iterations_before_retrain <= 0 or pd.isna(models_over_time[index-1]):

            train_window_end = index - 1
            
            X_expanding_window = X[first_nonzero_return:train_window_end]
            y_expanding_window = y[first_nonzero_return:train_window_end]

            current_transformations = [t.clone() for t in transformations]
            for transformation_index, transformation in enumerate(current_transformations):
                transformation.fit_transform(X_expanding_window, y_expanding_window)

            X_slice = X[train_window_start:train_window_end]

            for transformation in current_transformations:
                X_slice = transformation.transform(X_slice)
            
            X_slice = X_slice.to_numpy()
            y_slice = y[train_window_start:train_window_end].to_numpy()

            current_model = model.clone()

            current_model.initialize_network(input_dim = len(X_slice[0]), output_dim=1) 
            current_model.fit(X_slice, y_slice)

            iterations_before_retrain = retrain_every

        models_over_time[index] = current_model
        for transformation_index, transformation in enumerate(current_transformations):
            transformations_over_time[transformation_index][index] = transformation

        iterations_before_retrain -= 1

    return models_over_time, transformations_over_time

def walk_forward_inference(
                    model_name: str,
                    model_over_time: pd.Series,
                    transformations_over_time: list[pd.Series],
                    X: pd.DataFrame,
                    expanding_window: bool,
                    window_size: int,
                ) -> tuple[pd.Series, pd.DataFrame]:
    predictions = pd.Series(index=X.index).rename(model_name)
    probabilities = pd.DataFrame(index=X.index)

    inference_from = get_first_valid_return_index(model_over_time)
    inference_till = X.shape[0]
    first_model = model_over_time[inference_from]

    if first_model.only_column is not None:
        X = X[[column for column in X.columns if first_model.only_column in column]]
    
    if first_model.data_transformation == 'original':
        transformations_over_time = []

    for index in tqdm(range(inference_from, inference_till)):
        
        train_window_start = inference_from if expanding_window else index - window_size - 1

        current_model = model_over_time[index]
        current_transformations = [transformation_over_time[index] for transformation_over_time in transformations_over_time]

        if current_model.predict_window_size == 'window_size': 
            next_timestep = X.iloc[train_window_start:index]
        else: 
            # we need to get a Dataframe out of it, since the transformation step always expects a 2D array, but it's equivalent to X.iloc[index]
            next_timestep = X.iloc[index:index+1]
        
        for transformation in current_transformations:
            next_timestep = transformation.transform(next_timestep)

        next_timestep = next_timestep.to_numpy()

        prediction, probs = current_model.predict(next_timestep)
        predictions[index] = prediction
        if len(probabilities.columns) != len(probs):
            probabilities = probabilities.reindex(columns = ["prob_" + str(num) for num in range(0, len(probs.T))])
        probabilities.iloc[index] = probs

    return predictions, probabilities

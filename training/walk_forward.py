import pandas as pd
from models.base import Model
import numpy as np
from utils.helpers import get_first_valid_return_index
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from typing import Union
from sklearn.base import clone

def walk_forward_train(
                        model_name: str,
                        model: Model,
                        X: pd.DataFrame,
                        y: pd.Series,
                        target_returns: pd.Series,
                        expanding_window: bool,
                        window_size: int,
                        retrain_every: int,
                        scaler: Union[MinMaxScaler, Normalizer, StandardScaler],
                    ) -> tuple[pd.Series, pd.Series]:
    assert len(X) == len(y)
    models = pd.Series(index=y.index).rename(model_name)
    scalers = pd.Series(index=y.index).rename("scaler_" + model_name)

    first_nonzero_return = max(get_first_valid_return_index(target_returns), get_first_valid_return_index(X.iloc[:,0]), get_first_valid_return_index(y))
    train_from = first_nonzero_return + window_size + 1
    train_till = len(y)
    iterations_before_retrain = 0

    if model.only_column is not None:
        X = X[[column for column in X.columns if model.only_column in column]]
        
    is_scaling_on = model.data_scaling == 'scaled'

    if is_scaling_on:
        scaler = clone(scaler)

    for index in tqdm(range(train_from, train_till)):
        if expanding_window:
            train_window_start = first_nonzero_return
        else:
            train_window_start = index - window_size - 1

        if iterations_before_retrain <= 0 or pd.isna(models[index-1]):

            train_window_end = index - 1
            
            current_scaler = None
            if is_scaling_on:
                # We need to fit on the expanding window data slice
                # This is our only way to avoid lookahead bias
                current_scaler = clone(scaler)
                X_expanding_window = X[first_nonzero_return:train_window_end]
                current_scaler.fit(X_expanding_window.values)

            X_slice = X[train_window_start:train_window_end].to_numpy()
            y_slice = y[train_window_start:train_window_end].to_numpy()

            if is_scaling_on:
                X_slice = current_scaler.transform(X_slice)
            
            current_model = model.clone()

            current_model.initialize_network(input_dim = len(X_slice[0]), output_dim=1) 
            current_model.fit(X_slice, y_slice)

            iterations_before_retrain = retrain_every

        models[index] = current_model
        scalers[index] = current_scaler

        iterations_before_retrain -= 1

    return models, scalers

def walk_forward_inference(
                    model_name: str,
                    models: pd.Series,
                    scalers: pd.Series,
                    X: pd.DataFrame,
                    expanding_window: bool,
                    window_size: int,
                ) -> tuple[pd.Series, pd.DataFrame]:
    predictions = pd.Series(index=X.index).rename(model_name)
    probabilities = pd.DataFrame(index=X.index)

    first_nonzero_return = get_first_valid_return_index(models)
    train_from = first_nonzero_return
    train_till = X.shape[0]
    first_model = models[first_nonzero_return]

    if first_model.only_column is not None:
        X = X[[column for column in X.columns if first_model.only_column in column]]

    is_scaling_on = first_model.data_scaling == 'scaled'

    for index in tqdm(range(train_from, train_till)):
        if expanding_window:
            train_window_start = first_nonzero_return
        else:
            train_window_start = index - window_size - 1

        current_model = models[index]
        curren_scaler = scalers[index]

        if current_model.predict_window_size == 'window_size': 
            next_timestep = X.iloc[train_window_start:index].to_numpy()#.reshape(1, -1)
        else: 
            next_timestep = X.iloc[index].to_numpy().reshape(1, -1)
        
        if is_scaling_on:
            next_timestep = curren_scaler.transform(next_timestep)

        prediction, probs = current_model.predict(next_timestep)
        predictions[index] = prediction
        if len(probabilities.columns) != len(probs):
            probabilities = probabilities.reindex(columns = ["prob_" + str(num) for num in range(0, len(probs.T))])
        probabilities.iloc[index] = probs

    return predictions, probabilities

import pandas as pd
from models.base import Model
import numpy as np
from utils.helpers import get_first_valid_return_index
from tqdm import tqdm

from sklearn.base import clone

def walk_forward_train_test(
                            model_name: str,
                            model: Model,
                            X: pd.DataFrame,
                            y: pd.Series,
                            target_returns: pd.Series,
                            expanding_window: bool,
                            window_size: int,
                            retrain_every: int,
                            scaler,
                        ) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    assert len(X) == len(y)
    predictions = pd.Series(index=y.index).rename(model_name)
    probabilities = pd.DataFrame(index=y.index)
    models = pd.Series(index=y.index).rename(model_name)

    first_nonzero_return = max(get_first_valid_return_index(target_returns), get_first_valid_return_index(X.iloc[:,0]))
    train_from = first_nonzero_return + window_size + 1
    train_till = len(y)
    iterations_before_retrain = 0

    if model.only_column is not None:
        X = X[[column for column in X.columns if model.only_column in column]]

    is_scaling_on = scaler is not None and model.data_scaling == 'scaled'

    if is_scaling_on:
        scaler = clone(scaler)

    for index in tqdm(range(train_from, train_till)):

        if iterations_before_retrain <= 0 or pd.isna(models[index-1]):
            if expanding_window:
                train_window_start = first_nonzero_return
            else:
                train_window_start = index - window_size - 1

            train_window_end = index - 1
            
            if is_scaling_on:
                # We need to fit on the expanding window data slice
                # This is our only way to avoid lookahead bias
                X_expanding_window = X[first_nonzero_return:train_window_end]
                scaler.fit(X_expanding_window.values)

            X_slice = X[train_window_start:train_window_end]
            y_slice = y[train_window_start:train_window_end]

            if is_scaling_on:
                X_slice = scaler.transform(X_slice.values)
            else:
                X_slice = X_slice.to_numpy()

            current_model = model.clone()
            current_model.fit(X_slice, y_slice.to_numpy())
            iterations_before_retrain = retrain_every
        else:
            current_model = models[index-1]

        models[index] = current_model

        next_timestep = X.iloc[index].to_numpy().reshape(1, -1)
        if is_scaling_on:
            next_timestep = scaler.transform(next_timestep)

        prediction, probs = current_model.predict(next_timestep)
        predictions[index] = prediction
        if len(probabilities.columns) != len(probs):
            probabilities = probabilities.reindex(columns = ["prob_" + str(num) for num in range(0, len(probs.T))])
        probabilities.iloc[index] = probs

        iterations_before_retrain -= 1

    return models, predictions, probabilities

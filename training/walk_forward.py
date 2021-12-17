import pandas as pd
from sklearn.base import clone
from utils.typing import SKLearnModel
import numpy as np
from utils.helpers import get_first_valid_return_index

def walk_forward_train_test(
                            model_name: str,
                            model: SKLearnModel,
                            X: pd.DataFrame,
                            y: pd.Series,
                            target_returns: pd.Series,
                            window_size: int,
                            retrain_every: int,
                            scaler,
                        ) -> tuple[pd.Series, pd.Series]:
    assert len(X) == len(y)
    predictions = pd.Series(index=y.index).rename(model_name)
    models = pd.Series(index=y.index).rename(model_name)

    first_nonzero_return = max(get_first_valid_return_index(target_returns), get_first_valid_return_index(X.iloc[:,0]))
    train_from = first_nonzero_return + window_size + 1
    train_till = len(y)
    
    iterations_since_retrain = 0
    if scaler is not None:
        scaler = clone(scaler)

    for index in range(train_from, train_till):

        iterations_since_retrain += 1

        if iterations_since_retrain >= retrain_every or pd.isna(models[index-1]):
            train_window_start = index - window_size - 1
            train_window_end = index - 1
            
            if scaler is not None:
                # First we need to fit on the expanding window data slice
                # This is our only way to avoid lookahead bia
                X_expanding_window = X[first_nonzero_return:train_window_end]
                scaler.fit(X_expanding_window)

            X_slice = X[train_window_start:train_window_end]
            y_slice = y[train_window_start:train_window_end]

            if scaler is not None:
                X_slice = scaler.transform(X_slice)
            else:
                X_slice = X_slice.to_numpy()

            current_model = clone(model)
            current_model.fit(X_slice, y_slice.to_numpy())
            iterations_since_retrain = 0
        else:
            current_model = models[index-1]

        models[index] = current_model

        next_timestep = X.iloc[index].to_numpy().reshape(1, -1)
        prediction = current_model.predict(next_timestep).item()
        predictions[index] = prediction

    return models, predictions

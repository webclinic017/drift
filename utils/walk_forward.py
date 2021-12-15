import pandas as pd
from sklearn.base import clone
from utils.typing import SKLearnModel
import numpy as np

def walk_forward_train_test(
                            model_name: str,
                            model: SKLearnModel,
                            X: pd.DataFrame,
                            y: pd.Series,
                            window_size: int,
                            retrain_every: int
                        ) -> tuple[pd.Series, pd.Series]:
                        
    predictions = pd.Series(index=y.index).rename(model_name)
    models = pd.Series(index=y.index).rename(model_name)

    train_from = window_size
    train_till = y.index[-1]
    
    iterations_since_retrain = 0

    for i in range(train_from, train_till):

        iterations_since_retrain += 1
        window_start = i - window_size
        window_end = i
        X_slice = X[window_start:window_end]
        y_slice = y[window_start:window_end]

        if iterations_since_retrain >= retrain_every or pd.isna(models[i-1]):
            current_model = clone(model)
            current_model.fit(X_slice.to_numpy(), y_slice.to_numpy())
            iterations_since_retrain = 0
        else:
            current_model = models[i-1]

        models[window_end] = current_model

        next_timestep = X.iloc[window_end+1].to_numpy().reshape(1, -1)
        prediction = current_model.predict(next_timestep).item()
        if prediction == 0.:
            # TODO: we shouldn't feed in zeros to the model, and skip training / predicting when everything is 0
            # print("Warning: model predicted 0., overriding it with 0.0001")
            prediction = 0.0001
        predictions[window_end+1] = prediction

    return models, predictions

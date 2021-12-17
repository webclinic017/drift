import pandas as pd
from typing import Literal
from training.walk_forward import walk_forward_train_test
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from utils.evaluate import evaluate_predictions
from utils.typing import SKLearnModel

def __get_scaler(type: Literal['normalize', 'minmax', 'standardize', 'none']):
    if type == 'normalize':
        return Normalizer()
    elif type == 'minmax':
        return MinMaxScaler(feature_range= (-1, 1))
    elif type == 'standardize':
        return StandardScaler()
    else:
        return None

def run_single_asset_trainig_pipeline(
                    ticker_to_predict: str,
                    X: pd.DataFrame,
                    y: pd.Series,
                    target_returns: pd.Series,
                    models: list[tuple[str, SKLearnModel]],
                    method: Literal['regression', 'classification'],
                    sliding_window_size: int,
                    retrain_every: int,
                    scaler: Literal['normalize', 'minmax', 'standardize', 'none'],
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:


    scaler = __get_scaler(scaler)

    results = pd.DataFrame()
    predictions = pd.DataFrame()

    for model_name, model in models:

        model_over_time, preds = walk_forward_train_test(
            model_name=model_name,
            model = model,
            X = X,
            y = y,
            target_returns = target_returns,
            window_size = sliding_window_size,
            retrain_every = retrain_every,
            scaler = scaler
        )
        assert len(preds) == len(y)
        result = evaluate_predictions(
            model_name = model_name,
            target_returns = target_returns,
            y_pred = preds,
            sliding_window_size = sliding_window_size,
            method = method,
        )
        column_name = ticker_to_predict + "_" + model_name
        results[column_name] = result
        predictions[column_name] = preds

    return results, predictions
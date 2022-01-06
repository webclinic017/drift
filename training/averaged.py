
import pandas as pd
from utils.evaluate import evaluate_predictions

def average_and_evaluate_predictions(predictions: pd.DataFrame, y: pd.Series, target_returns: pd.Series, data_config: dict) -> tuple[pd.Series, pd.DataFrame]:
    averaged_predictions = predictions.mean(axis = 1)
    non_discretized_result = evaluate_predictions(
        model_name = 'Averaged - Non-discrete',
        target_returns = target_returns,
        y_pred = averaged_predictions,
        y_true = y,
        method = 'classification',
        no_of_classes = data_config['no_of_classes'],
        discretize=False
    )
    discretized_result = evaluate_predictions(
        model_name = 'Averaged - Discrete',
        target_returns = target_returns,
        y_pred = averaged_predictions,
        y_true = y,
        method = 'classification',
        no_of_classes = data_config['no_of_classes'],
        discretize=True
    )
    return averaged_predictions, pd.concat([non_discretized_result, discretized_result], axis = 1)
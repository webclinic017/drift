from utils.evaluate import discretize_threeway_threshold, evaluate_predictions
from utils.helpers import equal_except_nan
from training.primary_model import train_primary_model
import pandas as pd
from models.base import Model
from reporting.types import Reporting
from typing import Union, Optional
from config.config import Config

def train_meta_labeling_model(
                            target_asset: str,
                            X: pd.DataFrame,
                            input_predictions: pd.Series,
                            y: pd.Series,
                            target_returns: pd.Series,
                            models: list[tuple[str, Model]],
                            config: Config,
                            model_suffix: str,
                            from_index: Optional[int],
                            preloaded_models: Optional[list[tuple[str, pd.Series, list[pd.Series]]]] = None
                        ) -> tuple[pd.Series, pd.Series, pd.DataFrame, list[Reporting.Single_Model]]:

    discretize = discretize_threeway_threshold(0.33)
    discretized_predictions = input_predictions.apply(discretize)
    meta_y: pd.Series = pd.concat([discretized_predictions, y], axis=1).apply(equal_except_nan, axis = 1)

    meta_X = pd.concat([X, input_predictions, discretized_predictions], axis = 1)

    _, meta_preds, meta_probabilities, all_models_single_asset = train_primary_model(
        ticker_to_predict = "prediction_correct",
        X = meta_X,
        y = meta_y,
        target_returns = target_returns,
        models = models,
        expanding_window = config.expanding_window_meta_labeling,
        sliding_window_size = config.sliding_window_size_meta_labeling,
        retrain_every = config.retrain_every,
        from_index = from_index,
        scaler = config.scaler,
        no_of_classes = 'two',
        level = 'meta_labeling',
        print_results = False,
        preloaded_models = preloaded_models
    )
    if len(models) > 1:
        meta_preds = meta_preds.mean(axis = 1)
        bet_size = meta_probabilities[meta_probabilities.columns[1::2]].mean(axis = 1)
    else:
        bet_size = meta_probabilities.iloc[:,1]
    avg_predictions_with_sizing = input_predictions * bet_size
    avg_predictions_with_sizing.rename("model_" + target_asset + "_" + model_suffix, inplace=True)

    meta_result = evaluate_predictions(
        model_name = "Meta",
        target_returns = target_returns,
        y_pred = avg_predictions_with_sizing,
        y_true = y,
        no_of_classes = 'two',
        print_results = True,
        discretize=False
    )
    meta_result.rename("model_" + target_asset + "_" + model_suffix, inplace=True)
    

    return meta_result, avg_predictions_with_sizing, meta_probabilities, all_models_single_asset

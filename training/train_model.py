import pandas as pd
from typing import Literal, Optional
from training.walk_forward import walk_forward_train, walk_forward_inference
from utils.evaluate import evaluate_predictions
from models.base import Model
from .types import ModelOverTime, TransformationsOverTime, TrainingOutcome

def train_models(
                ticker_to_predict: str,
                X: pd.DataFrame,
                y: pd.Series,
                forward_returns: pd.Series,
                models: list[Model],
                expanding_window: bool,
                sliding_window_size: int,
                retrain_every: int,
                from_index: Optional[pd.Timestamp],
                no_of_classes: Literal['two', 'three-balanced', 'three-imbalanced'],
                level: str,
                print_results: bool,
                transformations_over_time: TransformationsOverTime,
                models_over_time: Optional[list[ModelOverTime]]
    ) -> list[TrainingOutcome]:
    return [train_model(ticker_to_predict, X, y, forward_returns, model, expanding_window, sliding_window_size, retrain_every, from_index, no_of_classes, level, print_results, transformations_over_time, models_over_time[index] if models_over_time else None) for index, model in enumerate(models)]


def train_model(
                ticker_to_predict: str,
                X: pd.DataFrame,
                y: pd.Series,
                forward_returns: pd.Series,
                model: Model,
                expanding_window: bool,
                sliding_window_size: int,
                retrain_every: int,
                from_index: Optional[pd.Timestamp],
                no_of_classes: Literal['two', 'three-balanced', 'three-imbalanced'],
                level: str,
                print_results: bool,
                transformations_over_time: TransformationsOverTime,
                model_over_time: Optional[ModelOverTime]
    ) -> TrainingOutcome:

    if model_over_time is None:
        print("Train model")
        model_over_time = walk_forward_train(
            model = model,
            X = X,
            y = y,
            forward_returns = forward_returns,
            expanding_window = expanding_window,
            window_size = sliding_window_size,
            retrain_every = retrain_every,
            from_index = from_index,
            transformations_over_time = transformations_over_time,
        )

    levelname = ("_" + level) if level == 'meta' else ""
    if model_over_time is None:
        model_id = "model_" + model.name + "_" + ticker_to_predict + levelname 
    else:
        model_id = model_over_time.name

    predictions, probabilities = walk_forward_inference(
        model_name = model_id,
        model_over_time= model_over_time,
        transformations_over_time = transformations_over_time,
        X = X,
        expanding_window = expanding_window,
        window_size = sliding_window_size,
        retrain_every = retrain_every,
        from_index = from_index,
    )
    
    assert len(predictions) == len(y)
    stats = evaluate_predictions(
        forward_returns = forward_returns,
        y_pred = predictions,
        y_true = y,
        no_of_classes=no_of_classes,
        print_results = print_results,
        discretize=True
    )

    return TrainingOutcome(model_id, predictions, probabilities, stats, model_over_time)
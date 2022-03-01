import pandas as pd
from typing import Literal, Optional
from training.walk_forward import (
    walk_forward_train,
    walk_forward_inference,
    walk_forward_inference_batched,
)
from utils.evaluate import evaluate_predictions
from models.base import Model
from .types import (
    ModelOverTime,
    TransformationsOverTime,
    TrainingOutcomeWithoutTransformations,
)


def train_model(
    ticker_to_predict: str,
    X: pd.DataFrame,
    y: pd.Series,
    forward_returns: pd.Series,
    model: Model,
    sliding_window_size: int,
    retrain_every: int,
    from_index: Optional[pd.Timestamp],
    no_of_classes: Literal["two", "three-balanced", "three-imbalanced"],
    level: str,
    output_stats: bool,
    transformations_over_time: TransformationsOverTime,
    model_over_time: Optional[ModelOverTime],
) -> TrainingOutcomeWithoutTransformations:

    if model_over_time is None:
        print("Train model")
        model_over_time = walk_forward_train(
            model=model,
            X=X,
            y=y,
            forward_returns=forward_returns,
            expanding_window=True,
            window_size=sliding_window_size,
            retrain_every=retrain_every,
            from_index=from_index,
            transformations_over_time=transformations_over_time,
        )

    levelname = ("_" + level) if level == "meta" else ""
    if model_over_time is None:
        model_id = "model_" + model.name + "_" + ticker_to_predict + levelname
    else:
        model_id = model_over_time.name

    inference_function = (
        walk_forward_inference
        if from_index is not None
        else walk_forward_inference_batched
    )
    predictions, probabilities = inference_function(
        model_name=model_id,
        model_over_time=model_over_time,
        transformations_over_time=transformations_over_time,
        X=X,
        expanding_window=True,
        window_size=sliding_window_size,
        retrain_every=retrain_every,
        from_index=from_index,
    )

    assert len(predictions) == len(y)
    if output_stats:
        stats = evaluate_predictions(
            forward_returns=forward_returns,
            y_pred=predictions,
            y_true=y,
            no_of_classes=no_of_classes,
            discretize=True,
        )
    else:
        stats = None

    return TrainingOutcomeWithoutTransformations(
        model_id, predictions, probabilities, stats, model_over_time
    )

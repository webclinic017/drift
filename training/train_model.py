import pandas as pd
from typing import Literal, Optional
from training.walk_forward import (
    walk_forward_train,
    walk_forward_inference,
    walk_forward_inference_batched,
)
from models.base import Model
from .types import (
    ModelOverTime,
    TransformationsOverTime,
    BaseTrainingOutcome,
)


def train_model(
    ticker_to_predict: str,
    X: pd.DataFrame,
    y: pd.Series,
    forward_returns: pd.Series,
    model: Model,
    initial_window_size: int,
    retrain_every: int,
    from_index: Optional[pd.Timestamp],
    level: str,
    class_labels: list[int],
    transformations_over_time: TransformationsOverTime,
    model_over_time: Optional[ModelOverTime],
) -> BaseTrainingOutcome:

    levelname = ("_" + level) if level == "meta" else ""
    model_id = (
        "model_" + model.name + "_" + ticker_to_predict + levelname
        if model_over_time is None
        else model_over_time.name
    )

    if model_over_time is None:
        print("Train model")
        model_over_time = walk_forward_train(
            model=model,
            X=X,
            y=y,
            forward_returns=forward_returns,
            window_size=initial_window_size,
            retrain_every=retrain_every,
            from_index=from_index,
            transformations_over_time=transformations_over_time,
        )

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
        window_size=initial_window_size,
        retrain_every=retrain_every,
        class_labels=class_labels,
        from_index=from_index,
    )

    assert len(predictions) == len(y)

    return BaseTrainingOutcome(model_id, predictions, probabilities, model_over_time)

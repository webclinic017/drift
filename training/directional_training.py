import pandas as pd

from transformations.base import Transformation

from .types import TrainingOutcome
from training.train_model import train_model
from training.walk_forward import walk_forward_process_transformations

from typing import Optional
from config.types import Config
from models.base import Model


def train_directional_model(
    X: pd.DataFrame,
    y: pd.Series,
    forward_returns: pd.Series,
    config: Config,
    model: Model,
    transformations: list[Transformation],
    from_index: Optional[pd.Timestamp],
    preloaded_training_step: Optional[TrainingOutcome] = None,
) -> TrainingOutcome:

    if preloaded_training_step is None:
        print("Preprocess transformations")
        transformations_over_time = walk_forward_process_transformations(
            X=X,
            y=y,
            forward_returns=forward_returns,
            window_size=config.sliding_window_size,
            retrain_every=config.retrain_every,
            from_index=from_index,
            transformations=transformations,
        )
    else:
        transformations_over_time = preloaded_training_step.transformations

    training_outcome = train_model(
        ticker_to_predict=config.target_asset[1],
        X=X,
        y=y,
        forward_returns=forward_returns,
        model=model,
        sliding_window_size=config.sliding_window_size,
        retrain_every=config.retrain_every,
        from_index=from_index,
        no_of_classes=config.no_of_classes,
        level="primary",
        output_stats=config.mode == "training",
        transformations_over_time=transformations_over_time,
        model_over_time=preloaded_training_step.model_over_time
        if preloaded_training_step
        else None,
    )
    if config.mode == "training":
        print(training_outcome.stats)

    return TrainingOutcome(
        **vars(training_outcome), transformations=transformations_over_time
    )

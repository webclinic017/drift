from data_loader.types import ForwardReturnSeries, XDataFrame, ySeries
from utils.evaluate import discretize_threeway_threshold, evaluate_predictions
from utils.helpers import equal_except_nan
from .train_model import train_model
import pandas as pd
from models.base import Model
from models.model_map import default_feature_selector_classification
from typing import Optional
from config.types import Config
from .types import BetSizingWithMetaOutcome, ModelOverTime, TransformationsOverTime
from training.walk_forward import walk_forward_process_transformations
from transformations.base import Transformation


def bet_sizing_with_meta_model(
    X: XDataFrame,
    input_predictions: pd.Series,
    y: ySeries,
    forward_returns: ForwardReturnSeries,
    model: Model,
    transformations: list[Transformation],
    config: Config,
    model_suffix: str,
    from_index: Optional[pd.Timestamp],
    transformations_over_time: Optional[TransformationsOverTime] = None,
    preloaded_models: Optional[ModelOverTime] = None,
) -> BetSizingWithMetaOutcome:

    input_predictions.name = "model_predictions"
    discretized_predictions = input_predictions.apply(
        discretize_threeway_threshold(0.33)
    )
    discretized_predictions.name = "model_discretized_predictions"

    meta_y: pd.Series = pd.concat([discretized_predictions, y], axis=1).apply(
        equal_except_nan, axis=1
    )
    meta_X = pd.concat([X, input_predictions, discretized_predictions], axis=1)

    if transformations_over_time is None:
        print("Preprocess transformations")
        transformations_over_time = walk_forward_process_transformations(
            X=meta_X,
            y=meta_y,
            forward_returns=forward_returns,
            window_size=config.sliding_window_size,
            retrain_every=config.retrain_every,
            from_index=from_index,
            transformations=transformations,
        )

    meta_outcome = train_model(
        ticker_to_predict="prediction_correct",
        X=meta_X,
        y=meta_y,
        forward_returns=forward_returns,
        model=model,
        sliding_window_size=config.sliding_window_size,
        retrain_every=config.retrain_every,
        from_index=from_index,
        no_of_classes="two",
        level="meta",
        output_stats=config.mode == "training",
        transformations_over_time=transformations_over_time,
        model_over_time=preloaded_models,
    )

    meta_predictions = meta_outcome.predictions
    bet_size = meta_outcome.probabilities.iloc[:, 1]
    avg_predictions_with_sizing = input_predictions * meta_predictions * bet_size

    if config.mode == "training":
        stats = evaluate_predictions(
            forward_returns=forward_returns,
            y_pred=avg_predictions_with_sizing,
            y_true=y,
            no_of_classes="three-balanced",
            discretize=False,
        )
        print(stats)
    else:
        stats = None
    model_id = "model_" + config.target_asset[1] + "_" + model_suffix

    return BetSizingWithMetaOutcome(
        model_id,
        meta_outcome,
        transformations_over_time,
        avg_predictions_with_sizing,
        stats,
    )

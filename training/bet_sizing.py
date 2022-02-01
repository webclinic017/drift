from data_loader.types import ForwardReturnSeries, XDataFrame, ySeries
from utils.evaluate import discretize_threeway_threshold, evaluate_predictions
from utils.helpers import equal_except_nan
from .train_model import train_models
import pandas as pd
from models.base import Model
from models.model_map import default_feature_selector_classification
from typing import Optional
from config.types import Config
from .types import BetSizingWithMetaOutcome, ModelOverTime, TransformationsOverTime
from training.walk_forward import walk_forward_process_transformations
from transformations.scaler import get_scaler
from transformations.rfe import RFETransformation
from transformations.pca import PCATransformation

def bet_sizing_with_meta_models(
                            X: XDataFrame,
                            input_predictions: pd.Series,
                            y: ySeries,
                            forward_returns: ForwardReturnSeries,
                            models: list[Model],
                            config: Config,
                            model_suffix: str,
                            from_index: Optional[pd.Timestamp],
                            transformations_over_time: Optional[TransformationsOverTime] = None,
                            preloaded_models: Optional[list[ModelOverTime]] = None
                        ) -> BetSizingWithMetaOutcome:

    input_predictions.name = "model_predictions"
    discretized_predictions = input_predictions.apply(discretize_threeway_threshold(0.33))
    discretized_predictions.name = "model_discretized_predictions"

    meta_y: pd.Series = pd.concat([discretized_predictions, y], axis=1).apply(equal_except_nan, axis = 1)
    meta_X = pd.concat([X, input_predictions, discretized_predictions], axis = 1)

    if transformations_over_time is None:
        print("Preprocess transformations")
        transformations_over_time = walk_forward_process_transformations(
            X = meta_X,
            y = meta_y,
            forward_returns = forward_returns,
            expanding_window = config.expanding_window_meta,
            window_size = config.sliding_window_size_meta,
            retrain_every = config.retrain_every,
            from_index = from_index,
            transformations= [
                get_scaler(config.scaler),
                PCATransformation(ratio_components_to_keep=0.5, sliding_window_size=config.sliding_window_size_meta),
                RFETransformation(n_feature_to_select=40, model=default_feature_selector_classification)
            ],
        )

    meta_outcomes = train_models(
        ticker_to_predict = "prediction_correct",
        X = meta_X,
        y = meta_y,
        forward_returns = forward_returns,
        models = models,
        expanding_window = config.expanding_window_meta,
        sliding_window_size = config.sliding_window_size_meta,
        retrain_every = config.retrain_every,
        from_index = from_index,
        no_of_classes = 'two',
        level = 'meta',
        output_stats = config.mode == 'training',
        transformations_over_time = transformations_over_time,
        models_over_time = preloaded_models,
    )

    # Ensemble predictions if necessary
    if len(models) > 1:
        meta_predictions = pd.concat([outcome.predictions for outcome in meta_outcomes], axis = 1).mean(axis = 1).apply(discretize_threeway_threshold(0.5))
        bet_size = pd.concat([outcome.probabilities[outcome.probabilities.columns[1::2]] for outcome in meta_outcomes], axis = 1).mean(axis = 1)
    else:
        meta_predictions = meta_outcomes[0].predictions
        bet_size = meta_outcomes[0].probabilities.iloc[:,1]
    avg_predictions_with_sizing = input_predictions * meta_predictions * bet_size

    if config.mode == 'training':
        stats = evaluate_predictions(
            forward_returns = forward_returns,
            y_pred = avg_predictions_with_sizing,
            y_true = y,
            no_of_classes = 'three-balanced',
            discretize=False
        )
        print(stats)
    else:
        stats = None
    model_id = "model_" + config.target_asset[1] + "_" + model_suffix

    return BetSizingWithMetaOutcome(model_id, meta_outcomes, transformations_over_time, avg_predictions_with_sizing, stats)

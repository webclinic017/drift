import pandas as pd

from .types import DirectionalTrainingOutcome, TrainingOutcome
from training.train_model import train_model
from training.walk_forward import walk_forward_process_transformations

from typing import Optional
from config.types import Config 
from models.base import Model
from models.model_map import default_feature_selector_classification

from transformations.scaler import get_scaler
from transformations.rfe import RFETransformation
from transformations.pca import PCATransformation

def train_directional_model(
                X: pd.DataFrame,
                y: pd.Series,
                forward_returns: pd.Series,
                config: Config,
                model: Model,
                from_index: Optional[pd.Timestamp],
                preloaded_training_step: Optional[DirectionalTrainingOutcome] = None,
            ) -> DirectionalTrainingOutcome:

    if preloaded_training_step is None:
        print("Preprocess transformations")
        transformations_over_time = walk_forward_process_transformations(
            X = X,
            y = y,
            forward_returns = forward_returns,
            expanding_window = config.expanding_window_base,
            window_size = config.sliding_window_size_base,
            retrain_every = config.retrain_every,
            from_index = from_index,
            transformations= [
                get_scaler(config.scaler),
                PCATransformation(ratio_components_to_keep=0.5, sliding_window_size=config.sliding_window_size_base),
                RFETransformation(n_feature_to_select=40, model=default_feature_selector_classification)
            ],
        )
    else:
        transformations_over_time = preloaded_training_step.transformations

    training_outcome = train_model(
        ticker_to_predict = config.target_asset[1],
        X = X,
        y = y,
        forward_returns = forward_returns,
        model = model,
        expanding_window = config.expanding_window_base,
        sliding_window_size = config.sliding_window_size_base,
        retrain_every = config.retrain_every,
        from_index = from_index,
        no_of_classes = config.no_of_classes,
        level = 'primary',
        output_stats= config.mode == 'training',
        transformations_over_time = transformations_over_time,
        model_over_time = preloaded_training_step.training.model_over_time if preloaded_training_step else None
    )
    if config.mode == 'training':
        print(training_outcome.stats)

    return DirectionalTrainingOutcome(training_outcome, transformations_over_time)


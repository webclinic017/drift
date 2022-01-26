from data_loader import load_data
from data_loader.process import check_data
from reporting.saving import load_models

from run_pipeline import run_pipeline
from config.types import Config, RawConfig
from config.presets import get_dev_config, get_default_ensemble_config, get_lightweight_ensemble_config
from labeling.process import label_data
from typing import Callable, Optional
from reporting.types import Reporting
from training.training_steps import primary_step, secondary_step
import pandas as pd
import warnings

def run_inference(preload_models:bool, raw_config: RawConfig):
    if preload_models:
        all_models, config = load_models(None)
    else:
        all_models, config, _, _, _ = run_pipeline(project_name='price-prediction', with_wandb = False, sweep = False, raw_config=raw_config)
    
    __inference(config, all_models.primary, all_models.secondary)


def __inference(config: Config, primary_models: Optional[Reporting.Training_Step], secondary_models: Optional[Reporting.Training_Step]):
    reporting = Reporting()
    asset = config.target_asset
    
    # 1. Load data, check for validity and process data
    X, returns, forward_returns = load_data(
        assets = config.assets,
        other_assets = config.other_assets,
        exogenous_data = config.exogenous_data,
        target_asset = config.target_asset,
        load_non_target_asset = config.load_non_target_asset,
        own_features = config.own_features,
        other_features = config.other_features,
        exogenous_features = config.exogenous_features,
    )
    assert check_data(X, config) == True, "Data is not valid. Cancelling Inference." 

    events, X, y, forward_returns = label_data(config.event_filter, config.labeling, X, returns, forward_returns)

    inference_from: pd.Timestamp = X.index[len(X.index) - 2]

    # 2. Train a Primary model with optional metalabeling for each asset
    training_step_primary, current_predictions = primary_step(X, y, forward_returns, config, reporting, from_index = inference_from, preloaded_training_step = primary_models)

    # 3. Train an Ensemble model with optional metalabeling for each asset
    if secondary_models is not None:
        warnings.warn("Secondary models are not specified.")
        training_step_secondary = secondary_step(X, y, current_predictions, forward_returns, config, reporting, from_index = inference_from, preloaded_training_step = secondary_models)

    # 4. Save the models
    reporting.asset = Reporting.Asset(name=asset[1], primary=training_step_primary, secondary=training_step_secondary)

    return reporting


if __name__ == '__main__':
    run_inference(preload_models=True, raw_config=get_lightweight_ensemble_config())
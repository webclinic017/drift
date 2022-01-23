from data_loader.load_data import load_data
from data_loader.process_data import check_data
from reporting.saving import load_models

from run_pipeline import run_pipeline
from config.config import Config, get_dev_config, get_default_ensemble_config, get_lightweight_ensemble_config

from typing import Callable, Optional
from reporting.types import Reporting
from training.training_steps import primary_step, secondary_step
import warnings

def run_inference(preload_models:bool, get_config:Callable):
    if preload_models:
        all_models, config = load_models(None)
    else:
        all_models, config, _, _, _ = run_pipeline(project_name='price-prediction', with_wandb = False, sweep = False, get_config=get_config)
    
    __inference(config, all_models.primary, all_models.secondary)


def __inference(config: Config, primary_models: Optional[Reporting.Training_Step], secondary_models: Optional[Reporting.Training_Step]):
    reporting = Reporting()
    asset = config.target_asset
    
    # 1. Load data, check for validity and process data
    X, y, target_returns = load_data(
    )
    assert check_data(X, y, config) == True, "Data is not valid. Cancelling Inference." 

    inference_from = X.index.stop - 2
    # 2. Train a Primary model with optional metalabeling for each asset
    training_step_primary, current_predictions = primary_step(X, y, target_returns, config, reporting, from_index = inference_from, preloaded_training_step = primary_models)

    # 3. Train an Ensemble model with optional metalabeling for each asset
    if secondary_models is not None:
        warnings.warn("Secondary models are not specified.")
        training_step_secondary = secondary_step(X, y, current_predictions, target_returns, config, reporting, from_index = inference_from, preloaded_training_step = secondary_models)

    # 4. Save the models
    reporting.asset = Reporting.Asset(ticker=asset, primary=training_step_primary, secondary=training_step_secondary)

    return reporting


if __name__ == '__main__':
    run_inference(preload_models=True, get_config=get_lightweight_ensemble_config)
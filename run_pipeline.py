import pandas as pd
from typing import Callable, Optional

from config.types import Config, RawConfig
from config.preprocess import preprocess_config, validate_config
from config.presets import get_default_ensemble_config, get_lightweight_ensemble_config

from data_loader.load import load_data
from data_loader.process import check_data

from labeling.process import label_data

from reporting.wandb import launch_wandb, override_config_with_wandb_values
from reporting.reporting import report_results

from reporting.saving import save_models
from reporting.types import Reporting

from training.training_steps import primary_step, secondary_step

import ray
ray.init()


def run_pipeline(project_name:str, with_wandb: bool, sweep: bool, raw_config: RawConfig) -> tuple[Reporting.Asset, Config, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wandb, config = __setup_config(project_name, with_wandb, sweep, raw_config)
    reporting = __run_training(config) 
    results, all_predictions, all_probabilities, all_models = reporting.get_results()
    report_results(results, all_predictions, config, wandb, sweep, project_name)
    save_models(all_models, config)

    return all_models, config, results, all_predictions, all_probabilities
    

def __setup_config(project_name:str, with_wandb: bool, sweep: bool, raw_config: RawConfig) -> tuple[Optional[object], Config]:
    wandb = None
    if with_wandb: 
        wandb = launch_wandb(project_name=project_name, default_config=raw_config, sweep=sweep)
        raw_config = override_config_with_wandb_values(wandb, raw_config)
    config = preprocess_config(raw_config)

    return wandb, config



def __run_training(config: Config):

    validate_config(config)
    reporting = Reporting()
    
    # 1. Load data, check for validity
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

    assert check_data(X, config) == True, "Data is not valid." 

    # 2. Filter for significant events when we want to trade, and label data
    events, X, y, forward_returns = label_data(config.event_filter, config.labeling, X, returns, forward_returns)

    # 3. Train a Primary model with optional metalabeling for each asset
    training_step_primary, current_predictions = primary_step(X, y, forward_returns, config, reporting, from_index = None)
    
    # 4. Train an Ensemble model with optional metalabeling for each asset
    training_step_secondary = secondary_step(X, y, current_predictions, forward_returns, config, reporting, from_index = None)
    
    # 5. Save the models
    reporting.asset = Reporting.Asset(name = config.target_asset[1], primary = training_step_primary, secondary = training_step_secondary)
    
    return reporting
    
    

    
if __name__ == '__main__':
    run_pipeline(project_name='price-prediction', with_wandb = False, sweep = False, raw_config=get_default_ensemble_config())
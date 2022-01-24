import pandas as pd
from typing import Callable, Optional

from data_loader.load_data import load_data
from data_loader.process_data import check_data

from reporting.wandb import launch_wandb, override_config_with_wandb_values
from reporting.reporting import report_results

from reporting.saving import save_models

from config.config import Config, get_default_ensemble_config, get_lightweight_ensemble_config
from config.preprocess import validate_config, preprocess_config

from training.training_steps import primary_step, secondary_step

from reporting.types import Reporting

import ray
ray.init()


def run_pipeline(project_name:str, with_wandb: bool, sweep: bool, get_config: Callable) -> tuple[Reporting.Asset, Config, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wandb, config = __setup_config(project_name, with_wandb, sweep, get_config)
    reporting = __run_training(config) 
    results, all_predictions, all_probabilities, all_models = reporting.get_results()
    report_results(results, all_predictions, config, wandb, sweep, project_name)
    save_models(all_models, config)

    return all_models, config, results, all_predictions, all_probabilities
    

def __setup_config(project_name:str, with_wandb: bool, sweep: bool, get_config: Callable) -> tuple[Optional[object], Config]:
    raw_config = get_config() 
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
    X, y, target_returns = load_data(
        assets = config.assets,
        other_assets = config.other_assets,
        exogenous_data = config.exogenous_data,
        target_asset = config.target_asset,
        load_non_target_asset = config.load_non_target_asset,
        log_returns = config.log_returns,
        forecasting_horizon = config.forecasting_horizon,
        own_features = config.own_features,
        other_features = config.other_features,
        exogenous_features = config.exogenous_features,
        no_of_classes = config.no_of_classes,
    )
    assert check_data(X, y, config) == True, "Data is not valid." 

    # 2. Train a Primary model with optional metalabeling for each asset
    training_step_primary, current_predictions = primary_step(X, y, target_returns, config, reporting, from_index = None)
    
    # 3. Train an Ensemble model with optional metalabeling for each asset
    training_step_secondary = secondary_step(X, y, current_predictions, target_returns, config, reporting, from_index = None)
    
    # 4. Save the models
    reporting.asset = Reporting.Asset(ticker= config.target_asset[1], primary=training_step_primary, secondary=training_step_secondary)
    
    return reporting
    
    

    
if __name__ == '__main__':
    run_pipeline(project_name='price-prediction', with_wandb = False, sweep = False, get_config=get_default_ensemble_config)
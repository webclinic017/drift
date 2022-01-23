import pandas as pd
from typing import Callable, Optional

from data_loader.load_data import load_data
from data_loader.process_data import check_data

from reporting.wandb import launch_wandb, register_config_with_wandb
from reporting.reporting import report_results

from reporting.saving import save_models

from config.config import get_default_ensemble_config
from config.preprocess import validate_config, preprocess_config

from training.training_steps import primary_step, secondary_step

from reporting.types import Reporting

import ray
ray.init()


def run_pipeline(project_name:str, with_wandb: bool, sweep: bool, get_config: Callable) -> tuple[Reporting.Asset, dict, dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wandb, model_config, training_config, data_config = __setup_config(project_name, with_wandb, sweep, get_config)
    reporting = __run_training(model_config, training_config, data_config) 
    results, all_predictions, all_probabilities, all_models = reporting.get_results()
    report_results(results, all_predictions, model_config, wandb, sweep, project_name)
    save_models(all_models, data_config, training_config, model_config)

    return all_models, data_config, training_config, model_config, results, all_predictions, all_probabilities
    

def __setup_config(project_name:str, with_wandb: bool, sweep: bool, get_config: Callable) -> tuple[Optional[object], dict, dict, dict]:
    model_config, training_config, data_config = get_config() 
    wandb = None
    if with_wandb: 
        wandb = launch_wandb(project_name=project_name, default_config=dict(**model_config, **training_config, **data_config), sweep=sweep)
        model_config, training_config, data_config = register_config_with_wandb(wandb, model_config, training_config, data_config)
    model_config, training_config, data_config = preprocess_config(model_config, training_config, data_config)

    return wandb, model_config, training_config, data_config



def __run_training(model_config:dict, training_config:dict, data_config:dict):

    validate_config(model_config, training_config, data_config)
    configs = dict(model_config=model_config, training_config=training_config, data_config=data_config)
    reporting = Reporting()
    
    # 1. Load data, check for validity
    X, y, target_returns = load_data(**configs['data_config'])
    assert check_data(X, y, configs['training_config']) == True, "Data is not valid." 

    # 2. Train a Primary model with optional metalabeling for each asset
    training_step_primary, current_predictions = primary_step(X, y, target_returns, configs, reporting, from_index = None)
    
    # 3. Train an Ensemble model with optional metalabeling for each asset
    training_step_secondary = secondary_step(X, y, current_predictions, target_returns, configs, reporting, from_index = None)
    
    # 4. Save the models
    reporting.asset = Reporting.Asset(ticker= data_config['target_asset'][1], primary=training_step_primary, secondary=training_step_secondary)
    
    return reporting
    
    

    
if __name__ == '__main__':
    run_pipeline(project_name='price-prediction', with_wandb = False, sweep = False, get_config=get_default_ensemble_config)
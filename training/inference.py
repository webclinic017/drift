
import pandas as pd
from typing import Optional, Union
import warnings

from data_loader.load_data import load_data
from data_loader.process_data import process_data, check_data

from reporting.types import Reporting
from training.training_steps import primary_step, secondary_step

def run_inference_pipeline(data_config:dict, training_config:dict, model_config:dict, all_models_all_assets:list[Reporting.Asset]):
    configs = dict(model_config=model_config, training_config=training_config, data_config=data_config)
    configs['data_config']['target_asset'] = data_config['assets'][0]
    
    primary_models, secondary_models = __select_models(configs, all_models_all_assets) 
    
    result = __inference(configs, primary_models, secondary_models)
    
    return result


def __inference(configs:dict, primary_models:Union[Reporting.Training_Step,None], secondary_models:Union[Reporting.Training_Step, None]):
    reporting = Reporting()
    asset = configs['data_config']['target_asset']
    
    # 1. Load data, truncate it, check for validity and process data (feature selection, dimensionality reduction, etc.)
    X, y, target_returns = load_data(**configs['data_config'])
    X, y  = __select_data(X, y, configs['training_config'])
    assert check_data(X, y, configs['training_config']) == False, "Data is not valid. Cancelling Inference." 
    X, original_X = process_data(X, y, configs)

    # 2. Train a Primary model with optional metalabeling for each asset
    training_step_primary, current_predictions = primary_step(X, y, original_X, asset, target_returns, configs, reporting, primary_models)

    # 3. Train an Ensemble model with optional metalabeling for each asset
    if secondary_step is not None:
        warnings.warn("Secondary models are not specified.")
        training_step_secondary = secondary_step(X, y, original_X, current_predictions, asset, target_returns, configs, reporting, secondary_models)

    # 4. Save the models
    reporting.all_assets.append(Reporting.Asset(ticker=asset, primary=training_step_primary, secondary=training_step_secondary))

    return reporting


def __select_models( configs:dict, all_models_all_assets:list[Reporting.Asset])-> tuple[Union[Reporting.Training_Step,None], Union[Reporting.Training_Step, None]]:
    target_asset_name = configs['data_config']['target_asset'][1]
    primary_step, secondary_step, = None, None
    target_asset_models = next((x for x in all_models_all_assets if x.name == target_asset_name), None)

    if target_asset_models is not None:
        if len(target_asset_models.primary.base)>0:
            primary_step = target_asset_models.primary
        else: warnings.warn("No primary models found for {}.".format(target_asset_name))

        if len(target_asset_models.secondary.base)>0:
            secondary_step = target_asset_models.secondary
        else: warnings.warn("No secondary models found for {}.".format(target_asset_name))
    else: 
        assert("No models found for asset: " + target_asset_name)
    
    return primary_step, secondary_step


def __select_data(X:pd.DataFrame, y:pd.Series, training_config:dict)-> tuple[pd.DataFrame, pd.Series]:
    window_size = training_config['sliding_window_size_primary']
    num_rows = X.shape[0]
    
    if num_rows <= window_size: 
        return X.copy(), y.copy() 
    else: 
        return X.truncate(before=int(num_rows-window_size), after=num_rows, copy=True), y.truncate(before=int(num_rows-window_size), after=num_rows, copy=True)



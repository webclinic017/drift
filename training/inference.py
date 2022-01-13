
import pandas as pd
from data_loader.load_data import load_data
from typing import Optional, Union
import warnings

from reporting.types import Reporting 

def run_inference_pipeline(data_config:dict, training_config:dict, all_models_all_assets:list[Reporting.Asset]):
    data_params = data_config.copy()
    data_params['target_asset'] = data_params['assets'][0]
    
    X, y, _ = load_data(**data_params)
    input_features = __select_data(X, training_config)
    
    primary_step, secondary_step = __select_models(data_params, all_models_all_assets) 
    
    result = __inference(input_features, primary_step, secondary_step)
    
    return result


def __inference(data:pd.DataFrame, primary_step:Union[Reporting.Training_Step,None], secondary_step:Union[Reporting.Training_Step,None]) -> pd.DataFrame:
    assert primary_step is not None, "No primary models found. Cancelling Inference."
    
    data = __primary_models(data, primary_step)
    
    if secondary_step is not None:
        warnings.warn("Secondary models are not specified.")
        data = __secondary_models(data, secondary_step)
    
    return data


def __select_models( data_params:dict, all_models_all_assets:list[Reporting.Asset])-> tuple[Union[Reporting.Training_Step,None], Union[Reporting.Training_Step, None]]:
    target_asset_name = data_params['target_asset'][1]
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


def __select_data(X:pd.DataFrame, training_config:dict)-> pd.DataFrame:
    window_size = training_config['sliding_window_size_primary']
    num_rows = X.shape[0]
    
    if num_rows <= window_size: 
        return X.copy() 
    else: 
        return X.truncate(before=int(num_rows-window_size), after=num_rows, copy=True)


def __primary_models(data:pd.DataFrame, models:dict)-> pd.DataFrame:
    for k, model in models:
        last_model = model[-1]
        prediction = last_model.predict(data.to_numpy())
        
        # result = evaluate_predictions(
        #     model_name = model_name,
        #     target_returns = target_returns,
        #     y_pred = preds,
        #     y_true = y,
        #     method = method,
        #     no_of_classes=no_of_classes,
        #     print_results = print_results,
        #     discretize=True
        # )
        
    return data

def __secondary_models(data:pd.DataFrame, model:dict)-> pd.DataFrame:
    
    return data

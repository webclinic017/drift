
import pandas as pd
from data_loader.load_data import load_data
from typing import Optional, Union
import warnings


def run_inference_pipeline(data_config:dict, training_config:dict, all_models_all_assets:dict):
    data_params = data_config.copy()
    data_params['target_asset'] = data_params['assets'][0]
    
    X, y, _ = load_data(**data_params)
    input_features = __select_data(X, training_config)
    
    primary_models, secondary_models = __select_models(data_params, all_models_all_assets) 
    
    result = __inference(input_features, primary_models, secondary_models)
    
    return result


def __inference(data:pd.DataFrame, primary_models:Union[dict,None], secondary_models:Union[dict,None]) -> pd.DataFrame:
    assert primary_models is not None, "No primary models found. Cancelling Inference."
    
    data = __primary_models(data, primary_models)
    
    if secondary_models is not None:
        warnings.warn("Secondary models are not specified.")
        data = __secondary_models(data, secondary_models)
    
    return data


def __select_models( data_params:dict, all_models_all_assets:dict)-> tuple[Optional[dict],Optional[dict]]:
    target_asset_name = data_params['target_asset'][1]
    primary_models, secondary_models = None, None

    if 'primary_models' in all_models_all_assets[target_asset_name]:
        primary_models = all_models_all_assets[target_asset_name]['primary_models']
    else: 
        assert("No primary models found for asset: " + target_asset_name)
    
    if 'secondary_model' in all_models_all_assets[target_asset_name]:
        secondary_models = all_models_all_assets[target_asset_name]['secondary_model']
    
    return primary_models, secondary_models


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

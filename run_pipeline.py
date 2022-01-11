from config.hashing import hash_data_config
from data_loader.load_data import load_data
import pandas as pd
from training.primary_model import train_primary_model
from reporting.wandb import launch_wandb, register_config_with_wandb
from models.model_map import default_feature_selector_regression, default_feature_selector_classification
from models.saving import save_models
from utils.helpers import has_enough_samples_to_train
from config.config import get_default_ensemble_config
from config.preprocess import validate_config, preprocess_config
from feature_selection.feature_selection import select_features
from feature_selection.dim_reduction import reduce_dimensionality
from training.meta_labeling import train_meta_labeling_model

from reporting.reporting import report_results
from typing import Callable, Optional
import ray
ray.init()


def run_pipeline(project_name:str, with_wandb: bool, sweep: bool, get_config: Callable) -> tuple[dict, dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wandb, model_config, training_config, data_config = __setup_config(project_name, with_wandb, sweep, get_config)
    results, all_predictions, all_probabilities, all_models_all_assets = __run_training(model_config, training_config, data_config)  
    report_results(results, all_predictions, model_config, wandb, sweep, project_name)
    save_models(all_models_all_assets, data_config, training_config)

    return all_models_all_assets, data_config, training_config, results, all_predictions, all_probabilities
    

def __setup_config(project_name:str, with_wandb: bool, sweep: bool, get_config: Callable) -> tuple[Optional[object], dict, dict, dict]:
    model_config, training_config, data_config = get_config() 
    wandb = None
    if with_wandb: 
        wandb = launch_wandb(project_name=project_name, default_config=dict(**model_config, **training_config, **data_config), sweep=sweep)
        model_config, training_config, data_config = register_config_with_wandb(wandb, model_config, training_config, data_config)
    model_config, training_config, data_config = preprocess_config(model_config, training_config, data_config)

    return wandb, model_config, training_config, data_config
    

def __run_training(model_config:dict, training_config:dict, data_config:dict):
    results = pd.DataFrame()
    all_predictions = pd.DataFrame()
    all_probabilities = pd.DataFrame()
    all_models_for_all_assets = dict()
    validate_config(model_config, training_config, data_config)

    for asset in data_config['assets']:
        print('--------\nPredicting: ', asset[1])

        # 1. Load data
        data_params = data_config.copy()
        data_params['target_asset'] = asset

        X, y, target_returns = load_data(**data_params)
        original_X = X.copy()
        if has_enough_samples_to_train(X, y, training_config) == False:
            print("Not enough samples to train")
            continue

        # 2a. Dimensionality Reduction (optional)
        if training_config['dimensionality_reduction']:
            X_pca = reduce_dimensionality(X, int(len(X.columns) / 2))
            X = X_pca.copy()
        else:
            X_pca = X.copy()

        # 2b. Feature Selection
        print("Feature Selection started")
        # TODO: this needs to be done per model!
        backup_model = default_feature_selector_regression if data_config['method'] == 'regression' else default_feature_selector_classification
        X = select_features(X = X, y = y, model = model_config['primary_models'][0][1], n_features_to_select = training_config['n_features_to_select'], backup_model = backup_model, scaling = training_config['scaler'])

        # 3. Train Primary models
        current_result, current_predictions, current_probabilities, all_models_for_single_asset = train_primary_model(
            ticker_to_predict = asset[1],
            original_X = original_X,
            X = X,
            y = y,
            target_returns = target_returns,
            models = model_config['primary_models'],
            method = data_config['method'],
            expanding_window = training_config['expanding_window_primary'],
            sliding_window_size = training_config['sliding_window_size_primary'],
            retrain_every =  training_config['retrain_every'],
            scaler =  training_config['scaler'],
            no_of_classes = data_config['no_of_classes'],
            level = 'primary',
            print_results= True
        )
        
        all_models_for_all_assets[asset[1]] = dict(
            name=asset[1], 
            primary_models = all_models_for_single_asset
        )
        
        # 4. Train a Meta-Labeling model for each Primary model and replace their predictions with the meta-labeling predictions
        if training_config['primary_models_meta_labeling'] == True:
            for model_name in current_result.columns:
                primary_model_predictions = current_predictions[model_name]
                primary_meta_result, primary_meta_preds, primary_meta_probabilities, meta_labeling_models = train_meta_labeling_model(
                    target_asset=asset[1],
                    X_pca = X_pca,
                    input_predictions= primary_model_predictions,
                    y = y,
                    target_returns = target_returns,
                    models = model_config['meta_labeling_models'],
                    data_config= data_config,
                    model_config= model_config,
                    training_config= training_config,
                    model_suffix = 'meta'
                )
                current_result[model_name] = primary_meta_result
                current_predictions[model_name] = primary_meta_preds

                all_models_for_all_assets[asset[1]]['primary_models'][model_name]['meta_labeling'] = meta_labeling_models
        
        results = pd.concat([results, current_result], axis=1)
        # With static models, because of the lag in the indicator, the first prediction is NA, so we fill it with zero.
        all_predictions = pd.concat([all_predictions, current_predictions], axis=1).fillna(0.)
        all_probabilities = pd.concat([all_probabilities, current_probabilities], axis=1).fillna(0.)

        # 5. Ensemble primary model predictions (If Ensemble model is present)
        if model_config['ensemble_model'] is not None:

            ensemble_result, ensemble_predictions, _, ensemble_models_one_asset = train_primary_model(
                ticker_to_predict = asset[1],
                original_X = current_predictions,
                X = current_predictions,
                y = y,
                target_returns = target_returns,
                models = [model_config['ensemble_model']],
                method = data_config['method'],
                expanding_window = False,
                sliding_window_size = 1,
                retrain_every = training_config['retrain_every'],
                scaler = training_config['scaler'],
                no_of_classes = data_config['no_of_classes'],
                level = 'ensemble',
                print_results= True,
            )
            ensemble_result, ensemble_predictions = ensemble_result.iloc[:,0], ensemble_predictions.iloc[:,0]
            all_models_for_all_assets[asset[1]]['secondary_model'] = ensemble_models_one_asset
            
            if len(model_config['meta_labeling_models']) > 0: 

                # 3. Train a Meta-labeling model on the averaged level-1 model predictions
                ensemble_meta_result, ensemble_meta_predictions, ensemble_meta_probabilities, ensemble_meta_labeling_models = train_meta_labeling_model(
                    target_asset=asset[1],
                    X_pca = X_pca,
                    input_predictions= ensemble_predictions,
                    y = y,
                    target_returns = target_returns,
                    models = model_config['meta_labeling_models'],
                    data_config= data_config,
                    model_config= model_config,
                    training_config= training_config,
                    model_suffix = 'ensemble'
                )
                
                all_models_for_all_assets[asset[1]]['secondary_model'][model_config['ensemble_model']] = dict(meta_labeling=ensemble_meta_labeling_models)
                results = pd.concat([results, ensemble_meta_result], axis=1)
                all_predictions = pd.concat([all_predictions, ensemble_meta_predictions], axis=1)
                all_probabilities = pd.concat([all_probabilities, ensemble_meta_probabilities], axis=1).fillna(0.)
                
    return results, all_predictions, all_probabilities, all_models_for_all_assets
    
    

    
if __name__ == '__main__':
    run_pipeline(project_name='price-prediction', with_wandb = False, sweep = False, get_config=get_default_ensemble_config)
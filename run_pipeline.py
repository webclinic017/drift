from config.hashing import hash_data_config
from data_loader.load_data import load_data
import pandas as pd
from training.training import run_single_asset_trainig
from reporting.wandb import launch_wandb, send_report_to_wandb, register_config_with_wandb
from models.model_map import default_feature_selector_regression, default_feature_selector_classification
from utils.helpers import get_first_valid_return_index
from config.config import get_default_level_1_daily_config, get_default_level_2_daily_config, get_default_level_2_hourly_config
from config.preprocess import validate_config, preprocess_config
from feature_selection.feature_selection import select_features
from feature_selection.dim_reduction import reduce_dimensionality
from training.meta_labeling import run_meta_labeling_training
from training.averaged import average_and_evaluate_predictions
from reporting.reporting import report_results
import ray
ray.init()


def run_pipeline(project_name:str, with_wandb: bool, sweep: bool, get_config:object):
    wandb, model_config, training_config, data_config = __setup_pipeline(project_name, with_wandb, sweep, get_config)
    results, all_predictions, all_probabilities = __run_training(model_config, training_config, data_config)  
    report_results(results, all_predictions, model_config, wandb, sweep, project_name)
    

def __setup_pipeline(project_name:str, with_wandb: bool, sweep: bool, get_config:object):
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
        first_valid_index = get_first_valid_return_index(X.iloc[:,0])
        samples_to_train = len(y) - first_valid_index
        if samples_to_train < training_config['sliding_window_size_level1'] * 3:
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
        X = select_features(X = X, y = y, model = model_config['level_1_models'][0][1], n_features_to_select = training_config['n_features_to_select'], backup_model = backup_model, scaling = training_config['scaler'], data_config_hash = hash_data_config(data_params))

        # 3. Train Level-1 models
        current_result, current_predictions, current_probabilities, all_models_for_single_asset = run_single_asset_trainig(
            ticker_to_predict = asset[1],
            original_X = original_X,
            X = X,
            y = y,
            target_returns = target_returns,
            models = model_config['level_1_models'],
            method = data_config['method'],
            expanding_window = training_config['expanding_window_level1'],
            sliding_window_size = training_config['sliding_window_size_level1'],
            retrain_every =  training_config['retrain_every'],
            scaler =  training_config['scaler'],
            no_of_classes = data_config['no_of_classes'],
            level = 1
        )
        
        all_models_for_all_assets[asset[1]] = dict(
            name=asset[1], 
            models=all_models_for_single_asset)
        
        # 4. Train a Meta-Labeling model for each Level-1 model and replace its predictions with the meta-labeling predictions
        if training_config['meta_labeling_lvl_1'] == True:
            for model_name in current_result.columns:
                lvl1_model_predictions = current_predictions[model_name]
                prev_sharpe = current_result[model_name]['sharpe']
                lvl1_meta_result, lvl1_meta_preds, lvl1_meta_probabilities, meta_labeling_models = run_meta_labeling_training(
                    target_asset=asset[1],
                    X_pca = X_pca,
                    input_predictions= lvl1_model_predictions,
                    y = y,
                    target_returns = target_returns,
                    data_config= data_config,
                    model_config= model_config,
                    training_config= training_config
                )
                new_sharpe = lvl1_meta_result['sharpe']
                print("Improvement in sharpe for the meta model: ", ((new_sharpe / prev_sharpe) - 1) * 100, "%")
                current_result[model_name] = lvl1_meta_result
                current_predictions[model_name] = lvl1_meta_preds

                all_models_for_all_assets[asset[1]][model_name] = meta_labeling_models
        
        results = pd.concat([results, current_result], axis=1)
        # With static models, because of the lag in the indicator, the first prediction is NA, so we fill it with zero.
        all_predictions = pd.concat([all_predictions, current_predictions], axis=1).fillna(0.)
        all_probabilities = pd.concat([all_probabilities, current_probabilities], axis=1).fillna(0.)

        if model_config['level_2_model'] is not None: 

            # 3. Average the Level-1 model predictions
            averaged_predictions, averaged_results = average_and_evaluate_predictions(current_predictions, y, target_returns, data_config)

            # 3. Train a Meta-labeling model on the averaged level-1 model predictions
            meta_result, avg_predictions_with_sizing, meta_probabilities, meta_labeling_models = run_meta_labeling_training(
                target_asset=asset[1],
                X_pca = X_pca,
                input_predictions= averaged_predictions,
                y = y,
                target_returns = target_returns,
                data_config= data_config,
                model_config= model_config,
                training_config= training_config
            )

            results = pd.concat([results, meta_result], axis=1)
            all_predictions = pd.concat([all_predictions, avg_predictions_with_sizing], axis=1)
            all_probabilities = pd.concat([all_probabilities, meta_probabilities], axis=1).fillna(0.)
        
    return results, all_predictions, all_probabilities
    
    

    
if __name__ == '__main__':
    run_pipeline(project_name='price-prediction', with_wandb = False, sweep = False, get_config=get_default_level_2_daily_config)
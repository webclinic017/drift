from config.hashing import hash_data_config
from data_loader.load_data import load_data
import pandas as pd
from training.training import run_single_asset_trainig
from reporting.wandb import launch_wandb, send_report_to_wandb, register_config_with_wandb
from models.model_map import default_feature_selector_regression, default_feature_selector_classification
from utils.helpers import get_first_valid_return_index, weighted_average
from config.config import get_default_level_1_daily_config, get_default_level_2_daily_config, get_default_level_2_hourly_config
from config.preprocess import validate_config, get_model_name, preprocess_config
from feature_selection.feature_selection import select_features
from feature_selection.dim_reduction import reduce_dimensionality
import ray
ray.init()


def run_pipeline(project_name:str, with_wandb: bool, sweep: bool):
    wandb, model_config, training_config, data_config = setup_pipeline(project_name, with_wandb, sweep)
    results, all_predictions, all_probabilities = run_training(project_name, wandb, sweep, model_config, training_config, data_config)  
    reporting(results, all_predictions, all_probabilities, model_config, wandb, sweep, project_name)
    

def setup_pipeline(project_name:str, with_wandb: bool, sweep: bool):
    model_config, training_config, data_config = get_default_level_2_daily_config()
    wandb = None
    if with_wandb: 
        wandb = launch_wandb(project_name=project_name, default_config=dict(**model_config, **training_config, **data_config), sweep=sweep)
        register_config_with_wandb(wandb, model_config, training_config, data_config)
    model_config, training_config, data_config = preprocess_config(model_config, training_config, data_config)


    return wandb, model_config, training_config, data_config
    





def run_training(project_name:str, wandb, sweep:bool, model_config:dict, training_config:dict, data_config:dict):
    results = pd.DataFrame()
    all_predictions = pd.DataFrame()
    all_probabilities = pd.DataFrame()
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
            X = reduce_dimensionality(X, int(len(X.columns) / 2))

        # 2b. Feature Selection (optional)
        if training_config['feature_selection']:
            print("Feature Selection started")
            # TODO: this needs to be done per model!
            backup_model = default_feature_selector_regression if data_config['method'] == 'regression' else default_feature_selector_classification
            X = select_features(X = X, y = y, model = model_config['level_1_models'][0][1], n_features_to_select = training_config['n_features_to_select'], backup_model = backup_model, scaling = training_config['scaler'], data_config_hash = hash_data_config(data_params))
            print("Feature Selection ended")

        # 3. Train Level-1 models
        current_result, current_predictions, current_probabilities = run_single_asset_trainig(
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
        results = pd.concat([results, current_result], axis=1)
        # With static models, because of the lag in the indicator, the first prediction is NA, so we fill it with zero.
        all_predictions = pd.concat([all_predictions, current_predictions], axis=1).fillna(0.)
        all_probabilities = pd.concat([all_probabilities, current_probabilities], axis=1).fillna(0.)

        # 3. Train Level-2 (Ensemble) model (Optional)
        if model_config['level_2_model'] is not None: 
            ensemble_X = pd.concat([all_predictions, all_probabilities], axis = 1)
            if training_config['include_original_data_in_ensemble']:
                ensemble_X = pd.concat([ensemble_X, X], axis=1)

            ensemble_result, ensemble_preds, ensemble_probabilities = run_single_asset_trainig(
                ticker_to_predict = asset[1],
                original_X = ensemble_X,
                X = ensemble_X,
                y = y,
                target_returns = target_returns,
                models = [model_config['level_2_model']],
                method = data_config['method'],
                expanding_window = training_config['expanding_window_level2'],
                sliding_window_size = training_config['sliding_window_size_level2'],
                retrain_every = training_config['retrain_every'],
                scaler = training_config['scaler'],
                no_of_classes = data_config['no_of_classes'],
                level = 2
            )

            results = pd.concat([results, ensemble_result], axis=1)
            all_predictions = pd.concat([all_predictions, ensemble_preds], axis=1)
            all_probabilities = pd.concat([all_probabilities, ensemble_probabilities], axis=1).fillna(0.)
        
    return results, all_predictions, all_probabilities
    
    
def reporting(results:pd.DataFrame, all_predictions:pd.DataFrame, all_probabilities:pd.DataFrame, model_config:dict, wandb, sweep: bool, project_name:str):
    results.to_csv('results.csv')

    level1_columns = results[[column for column in results.columns if 'lvl1' in column]]
    level2_columns = results[[column for column in results.columns if 'lvl2' in column]]

    # Only send the results of the final model to wandb
    results_to_send = level2_columns if level2_columns.shape[1] > 0 else level1_columns
    send_report_to_wandb(results_to_send, wandb, project_name, get_model_name(model_config))

    level1_predictions = all_predictions[[column for column in all_predictions.columns if 'lvl1' in column]]
    level2_predictions = all_predictions[[column for column in all_predictions.columns if 'lvl2' in column]]
    predictions_to_save = level2_predictions if level2_predictions.shape[1] > 0 else level1_predictions
    predictions_to_save.to_csv('predictions.csv')

    print("\n--------\n")
    print("Benchmark buy-and-hold sharpe: ", round(weighted_average(results, 'no_of_samples').loc['benchmark_sharpe'], 3))

    print("Level-1: Number of samples evaluated: ", level1_columns.loc['no_of_samples'].sum())
    print("Mean Sharpe ratio for Level-1 models: ", round(weighted_average(level1_columns, 'no_of_samples').loc['sharpe'], 3))
    print("Mean Probabilistic Sharpe ratio for Level-1 models: ", round(weighted_average(level1_columns, 'no_of_samples').loc['prob_sharpe'].mean(), 3))

    if model_config['level_2_model'] is not None: 
        print("Level-2 (Ensemble): Number of samples evaluated: ", level2_columns.loc['no_of_samples'].sum())
        print("Mean Sharpe ratio for Level-2 (Ensemble) models: ", round(weighted_average(level2_columns, 'no_of_samples').loc['sharpe'].mean(), 3))
        print("Mean Probabilistic Sharpe ratio for Level-2 (Ensemble) models: ", round(weighted_average(level2_columns, 'no_of_samples').loc['prob_sharpe'].mean(), 3))

    if sweep:
        if wandb.run is not None:
            wandb.finish()
    
if __name__ == '__main__':
    run_pipeline(project_name='price-prediction', with_wandb = False, sweep = False)
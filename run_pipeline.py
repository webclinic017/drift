from utils.load_data import load_data
import pandas as pd
from training.training import run_single_asset_trainig
from reporting.wandb import launch_wandb, send_report_to_wandb, register_config_with_wandb
from models.model_map import map_model_name_to_function
from feature_extractors.feature_extractor_presets import preprocess_feature_extractors_config
from config import get_default_config, validate_config, get_model_name

def setup_pipeline(project_name:str, with_wandb: bool, sweep: bool):
    model_config, training_config, data_config = get_default_config()
    
    wandb = None
    if with_wandb: 
        wandb = launch_wandb(project_name=project_name, default_config=dict(**model_config, **training_config, **data_config), sweep=sweep)
        register_config_with_wandb(wandb, model_config, training_config, data_config)

    model_config = map_model_name_to_function(model_config, data_config['method'])
    data_config = preprocess_feature_extractors_config(data_config)
    pipeline(project_name, wandb, sweep, model_config, training_config, data_config)  
    


def pipeline(project_name:str, wandb, sweep:bool, model_config:dict, training_config:dict, data_config:dict ):
    results = pd.DataFrame()
    validate_config(model_config, training_config, data_config)

    for asset in data_config['all_assets']:
        print('--------\nPredicting: ', asset)
        all_predictions = pd.DataFrame()

        # 1. Load data
        data_params = data_config.copy()
        data_params['target_asset'] = asset

        X, y, target_returns = load_data(**data_params)

        # 2. Train Level-1 models
        current_result, current_predictions = run_single_asset_trainig(
            ticker_to_predict = asset,
            X = X,
            y = y,
            target_returns = target_returns,
            models = model_config['level_1_models'],
            method = data_config['method'],
            expanding_window = training_config['expanding_window'],
            sliding_window_size = training_config['sliding_window_size'],
            retrain_every =  training_config['retrain_every'],
            scaler =  training_config['scaler'],
            no_of_classes = data_config['no_of_classes']
        )
        results = pd.concat([results, current_result], axis=1)
        all_predictions = pd.concat([all_predictions, current_predictions], axis=1)

        if len(model_config['level_2_models']) > 0: 
            # 3. Train Level-2 (Ensemble) model
            
            ensemble_X = all_predictions
            if training_config['include_original_data_in_ensemble']:
                ensemble_X = pd.concat([ensemble_X, X], axis=1)

            ensemble_result, ensemble_preds = run_single_asset_trainig(
                ticker_to_predict = asset,
                X = ensemble_X,
                y = y,
                target_returns = target_returns,
                models = model_config['level_2_models'],
                method = data_config['method'],
                expanding_window = training_config['expanding_window'],
                sliding_window_size = training_config['sliding_window_size'],
                retrain_every = training_config['retrain_every'],
                scaler = training_config['scaler'],
                no_of_classes = data_config['no_of_classes']
            )

            results = pd.concat([results, ensemble_result], axis=1)
            all_predictions = pd.concat([all_predictions, ensemble_preds], axis=1)
        
    # 4. Save & report results
    send_report_to_wandb(results, wandb, project_name, get_model_name(model_config))
    results.to_csv('results.csv')

    level1_columns = results[[column for column in results.columns if 'Ensemble' not in column]]
    ensemble_columns = results[[column for column in results.columns if 'Ensemble' in column]]

    print("Mean no of samples: ", results.loc['no_of_samples'].mean())
    print("Mean Sharpe ratio for Level-1 models: ", round(level1_columns.loc['sharpe'].mean(), 3))
    print("Mean Sharpe ratio for Level-2 (Ensemble) models: ", round(ensemble_columns.loc['sharpe'].mean(), 3))

    if sweep:
        if wandb.run is not None:
            wandb.finish()
    
if __name__ == '__main__':
    setup_pipeline(project_name='price-prediction', with_wandb = False, sweep = False)
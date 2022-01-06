from reporting.wandb import send_report_to_wandb
import pandas as pd
from config.preprocess import get_model_name
from utils.helpers import weighted_average

def report_results(results:pd.DataFrame, all_predictions:pd.DataFrame, model_config:dict, wandb, sweep: bool, project_name:str):

    level1_columns = results[[column for column in results.columns if 'lvl1' in column]]
    level2_columns = results[[column for column in results.columns if 'lvl2' in column]]

    # Only send the results of the final model to wandb
    results_to_send = level2_columns if level2_columns.shape[1] > 0 else level1_columns
    send_report_to_wandb(results_to_send, wandb, project_name, get_model_name(model_config))
    results.to_csv('results.csv')

    level1_predictions = all_predictions[[column for column in all_predictions.columns if 'lvl1' in column]]
    level2_predictions = all_predictions[[column for column in all_predictions.columns if 'lvl2' in column]]
    predictions_to_save = level2_predictions if level2_predictions.shape[1] > 0 else level1_predictions
    predictions_to_save.to_csv('predictions.csv')

    print("\n--------\n")
    all_avg_results = weighted_average(results, 'no_of_samples')
    lvl1_avg_results = weighted_average(level1_columns, 'no_of_samples')
    lvl2_avg_results = weighted_average(level2_columns, 'no_of_samples')

    print("Benchmark buy-and-hold sharpe: ", round(all_avg_results.loc['benchmark_sharpe'], 3))

    print("Level-1: Number of samples evaluated: ", level1_columns.loc['no_of_samples'].sum())
    print("Mean Sharpe ratio for Level-1 models: ", round(lvl1_avg_results.loc['sharpe'], 3))
    print("Mean Probabilistic Sharpe ratio for Level-1 models: ", round(lvl1_avg_results.loc['prob_sharpe'].mean(), 3))

    if model_config['level_2_model'] is not None: 
        print("Level-2 (Ensemble): Number of samples evaluated: ", level2_columns.loc['no_of_samples'].sum())
        print("Mean Sharpe ratio for Level-2 (Ensemble) models: ", round(lvl2_avg_results.loc['sharpe'].mean(), 3))
        print("Mean Probabilistic Sharpe ratio for Level-2 (Ensemble) models: ", round(lvl2_avg_results.loc['prob_sharpe'].mean(), 3))

    lvl2_avg_results.to_csv('results_level2.csv')

    if sweep:
        if wandb.run is not None:
            wandb.finish()
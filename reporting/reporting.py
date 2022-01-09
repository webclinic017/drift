from reporting.wandb import send_report_to_wandb
import pandas as pd
from config.preprocess import get_model_name
from utils.helpers import weighted_average

def report_results(results:pd.DataFrame, all_predictions:pd.DataFrame, model_config:dict, wandb, sweep: bool, project_name:str):

    primary_results = results[[column for column in results.columns if 'primary' in column]]
    ensemble_results = results[[column for column in results.columns if 'ensemble' in column]]

    # Only send the results of the final model to wandb
    results_to_send = ensemble_results if ensemble_results.shape[1] > 0 else primary_results
    send_report_to_wandb(results_to_send, wandb, project_name, get_model_name(model_config))
    results.to_csv('output/results.csv')

    primary_weights = all_predictions[[column for column in all_predictions.columns if 'primary' in column]]
    ensemble_weights = all_predictions[[column for column in all_predictions.columns if 'ensemble' in column]]
    predictions_to_save = ensemble_weights if ensemble_weights.shape[1] > 0 else primary_weights
    predictions_to_save.to_csv('output/predictions.csv')

    print("\n--------\n")
    all_avg_results = weighted_average(results, 'no_of_samples')
    primary_avg_results = weighted_average(primary_results, 'no_of_samples')
    ensemble_avg_results = weighted_average(ensemble_results, 'no_of_samples')

    print("Benchmark buy-and-hold sharpe: ", round(all_avg_results.loc['benchmark_sharpe'], 3))

    print("Level-1: Number of samples evaluated: ", primary_results.loc['no_of_samples'].sum())
    print("Mean Sharpe ratio for Level-1 models: ", round(primary_avg_results.loc['sharpe'], 3))
    print("Mean Probabilistic Sharpe ratio for Level-1 models: ", round(primary_avg_results.loc['prob_sharpe'].mean(), 3))

    if len(model_config['meta_labeling_models']) > 0: 
        print("Level-2 (Ensemble): Number of samples evaluated: ", ensemble_results.loc['no_of_samples'].sum())
        print("Mean Sharpe ratio for Level-2 (Ensemble) models: ", round(ensemble_avg_results.loc['sharpe'].mean(), 3))
        print("Mean Probabilistic Sharpe ratio for Level-2 (Ensemble) models: ", round(ensemble_avg_results.loc['prob_sharpe'].mean(), 3))

    ensemble_avg_results.to_csv('output/results_level2.csv')

    if sweep:
        if wandb.run is not None:
            wandb.finish()
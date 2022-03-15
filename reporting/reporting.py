from reporting.wandb import send_report_to_wandb
import pandas as pd
from utils.helpers import weighted_average
from config.types import Config
from training.types import WeightsSeries, Stats


def report_results(
    directional_stats: Stats,
    output_stats: Stats,
    output_weights: WeightsSeries,
    config: Config,
    wandb,
):

    # Only send the results of the final model to wandb
    send_report_to_wandb(output_stats, wandb)
    pd.Series(output_stats).to_csv("output/results.csv")

    output_weights.rename(config.target_asset.file_name).to_csv(
        "output/predictions.csv"
    )

    print("\n--------\n")

    print("Benchmark buy-and-hold sharpe: ", output_stats["benchmark_sharpe"])

    print("Level-1: Number of samples evaluated: ", directional_stats["no_of_samples"])
    print(
        "Mean Sharpe ratio for Level-1 models: ", round(directional_stats["sharpe"], 3)
    )
    print(
        "Mean Probabilistic Sharpe ratio for Level-1 models: ",
        round(directional_stats["prob_sharpe"], 3),
    )

    print(
        "Level-2 (Ensemble): Number of samples evaluated: ",
        output_stats["no_of_samples"],
    )
    print("Mean Sharpe ratio for Level-2 (Ensemble) models: ", output_stats["sharpe"])
    print(
        "Mean Probabilistic Sharpe ratio for Level-2 (Ensemble) models: ",
        output_stats["prob_sharpe"],
    )

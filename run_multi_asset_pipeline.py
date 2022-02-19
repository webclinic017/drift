from run_pipeline import run_training, setup_config
from config.types import RawConfig
from data_loader.collections import data_collections
from reporting.reporting import report_results
from config.presets import get_default_config


def run_multi_asset_pipeline(
    project_name: str, with_wandb: bool, sweep: bool, raw_config: RawConfig
):
    collection = data_collections["fivemin_crypto"]
    for asset in collection:
        print(f"# Predicting asset: {asset[1]}\n")
        raw_config.target_asset = asset[1]
        wandb, config = setup_config(project_name, with_wandb, sweep, raw_config)
        pipeline_outcome = run_training(config)
        report_results(
            pipeline_outcome.directional_training.training.stats,
            pipeline_outcome.get_output_stats(),
            pipeline_outcome.get_output_weights(),
            config,
            wandb,
            sweep,
        )


if __name__ == "__main__":
    run_multi_asset_pipeline(
        project_name="price-prediction",
        with_wandb=False,
        sweep=False,
        raw_config=get_default_config(),
    )

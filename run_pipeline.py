from typing import Optional

from config.types import Config, RawConfig
from config.preprocess import preprocess_config
from config.presets import get_default_ensemble_config, get_lightweight_ensemble_config

from data_loader.load import load_data
from data_loader.process import check_data

from labeling.process import label_data

from reporting.wandb import launch_wandb, override_config_with_wandb_values
from reporting.reporting import report_results
from reporting.saving import save_models

from training.directional_training import train_directional_model
from training.bet_sizing import bet_sizing_with_meta_model
from training.types import PipelineOutcome

import ray

ray.init()


def run_pipeline(
    project_name: str, with_wandb: bool, sweep: bool, raw_config: RawConfig
) -> tuple[PipelineOutcome, Config]:
    wandb, config = __setup_config(project_name, with_wandb, sweep, raw_config)
    pipeline_outcome = __run_training(config)
    report_results(
        pipeline_outcome.directional_training.training.stats,
        pipeline_outcome.get_output_stats(),
        pipeline_outcome.get_output_weights(),
        config,
        wandb,
        sweep,
    )
    save_models(pipeline_outcome, config)
    return pipeline_outcome, config


def __setup_config(
    project_name: str, with_wandb: bool, sweep: bool, raw_config: RawConfig
) -> tuple[Optional[object], Config]:
    wandb = None
    if with_wandb:
        wandb = launch_wandb(
            project_name=project_name, default_config=raw_config, sweep=sweep
        )
        raw_config = override_config_with_wandb_values(wandb, raw_config)
    config = preprocess_config(raw_config)

    return wandb, config


def __run_training(config: Config) -> PipelineOutcome:

    print("---> Load data, check for validity")
    X, returns = load_data(
        assets=config.assets,
        other_assets=config.other_assets,
        exogenous_data=config.exogenous_data,
        target_asset=config.target_asset,
        load_non_target_asset=config.load_non_target_asset,
        own_features=config.own_features,
        other_features=config.other_features,
        exogenous_features=config.exogenous_features,
    )

    assert check_data(X, config) == True, "Data is not valid."

    print("---> Filter for significant events when we want to trade, and label data")
    events, X, y, forward_returns = label_data(
        config.event_filter, config.labeling, X, returns
    )

    print("---> Train directional models")
    directional_training_outcome = train_directional_model(
        X,
        y,
        forward_returns,
        config,
        config.directional_model,
        config.transformations,
        from_index=None,
        preloaded_training_step=None,
    )

    print("---> Run bet sizing on directional model's output")
    bet_sizing_outcomes = bet_sizing_with_meta_model(
        X,
        directional_training_outcome.training.predictions,
        y,
        forward_returns,
        config.meta_model,
        config.transformations,
        config,
        "meta",
        None,
        None,
        None,
    )

    return PipelineOutcome(directional_training_outcome, bet_sizing_outcomes)


if __name__ == "__main__":
    run_pipeline(
        project_name="price-prediction",
        with_wandb=False,
        sweep=False,
        raw_config=get_lightweight_ensemble_config(),
    )

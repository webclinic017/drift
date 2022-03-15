from typing import Optional

from config.types import Config, RawConfig
from config.preprocess import preprocess_config
from config.presets import get_default_config

from data_loader.load import load_data
from data_loader.process import check_data

from labeling.process import label_data

from reporting.wandb import launch_wandb, override_config_with_wandb_values
from reporting.reporting import report_results
from reporting.saving import save_models

from training.directional_training import train_directional_model
from training.bet_sizing import bet_sizing_with_meta_model
from training.types import PipelineOutcome


def run_pipeline(
    project_name: str, with_wandb: bool, raw_config: RawConfig
) -> tuple[PipelineOutcome, Config]:
    wandb, config = setup_config(project_name, with_wandb, raw_config)
    outcome = run_training(config)
    report_results(
        outcome.directional_training.stats,
        outcome.get_output_stats(),
        outcome.get_output_weights(),
        config,
        wandb,
    )
    if config.save_models:
        save_models(outcome, config)
    return outcome, config


def setup_config(
    project_name: str, with_wandb: bool, raw_config: RawConfig
) -> tuple[Optional[object], Config]:
    wandb = None
    if with_wandb:
        wandb = launch_wandb(project_name=project_name, default_config=raw_config)
        raw_config = override_config_with_wandb_values(wandb, raw_config)
    config = preprocess_config(raw_config)

    return wandb, config


def run_training(config: Config) -> PipelineOutcome:

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
        start_date=config.start_date,
    )

    assert check_data(X, config) == True, "Data is not valid."

    print("---> Filter for significant events when we want to trade, and label data")
    events, X, y, forward_returns = label_data(
        event_filter=config.event_filter,
        event_labeller=config.labeling,
        X=X,
        returns=returns,
        remove_overlapping_events=config.remove_overlapping_events,
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
        directional_training_outcome.predictions,
        y,
        forward_returns,
        config.meta_model,
        config.transformations,
        config,
        None,
        None,
        None,
    )

    return PipelineOutcome(directional_training_outcome, bet_sizing_outcomes)


if __name__ == "__main__":
    run_pipeline(
        project_name="price-prediction",
        with_wandb=False,
        raw_config=get_default_config(),
    )

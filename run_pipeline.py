from typing import Optional

from config.types import Config, RawConfig
from config.preprocess import preprocess_config, validate_config
from config.presets import get_default_ensemble_config, get_lightweight_ensemble_config

from data_loader.load import load_data
from data_loader.process import check_data

from labeling.process import label_data

from reporting.wandb import launch_wandb, override_config_with_wandb_values
from reporting.reporting import report_results
from reporting.saving import save_models

from training.directional_training import train_directional_models
from training.bet_sizing import bet_sizing_with_meta_models
from training.ensemble import ensemble_weights
from training.types import PipelineOutcome

import ray
ray.init()


def run_pipeline(project_name:str, with_wandb: bool, sweep: bool, raw_config: RawConfig) -> tuple[PipelineOutcome, Config]:
    wandb, config = __setup_config(project_name, with_wandb, sweep, raw_config)
    pipeline_outcome = __run_training(config) 
    report_results([s.stats for s in pipeline_outcome.directional_training.training], pipeline_outcome.get_output_stats(), pipeline_outcome.get_output_weights(), config, wandb, sweep)
    save_models(pipeline_outcome, config)
    return pipeline_outcome, config
    

def __setup_config(project_name:str, with_wandb: bool, sweep: bool, raw_config: RawConfig) -> tuple[Optional[object], Config]:
    wandb = None
    if with_wandb: 
        wandb = launch_wandb(project_name=project_name, default_config=raw_config, sweep=sweep)
        raw_config = override_config_with_wandb_values(wandb, raw_config)
    config = preprocess_config(raw_config)

    return wandb, config



def __run_training(config: Config) -> PipelineOutcome:

    validate_config(config)
    
    # 1. Load data, check for validity
    X, returns, forward_returns = load_data(
        assets = config.assets,
        other_assets = config.other_assets,
        exogenous_data = config.exogenous_data,
        target_asset = config.target_asset,
        load_non_target_asset = config.load_non_target_asset,
        own_features = config.own_features,
        other_features = config.other_features,
        exogenous_features = config.exogenous_features,
    )

    assert check_data(X, config) == True, "Data is not valid." 

    # 2. Filter for significant events when we want to trade, and label data
    events, X, y, forward_returns = label_data(config.event_filter, config.labeling, X, returns, forward_returns)

    # 3. Train directional models
    directional_training_outcome = train_directional_models(X, y, forward_returns, config, config.directional_models, from_index = None, preloaded_training_step = None)
    
    # 4. Run bet sizing on primary model's output
    bet_sizing_outcomes = [bet_sizing_with_meta_models(X, outcome.predictions, y, forward_returns, config.meta_models, config, 'meta', None, None, None) for outcome in directional_training_outcome.training]

    # 4. Ensemble weights
    ensemble_outcome = ensemble_weights([o.weights for o in bet_sizing_outcomes], forward_returns, y, config.no_of_classes, config.mode == 'training')

    # 5. (Optional) Additional bet sizing on top of the ensembled weights
    ensemble_bet_sizing_outcome = bet_sizing_with_meta_models(X, ensemble_outcome.weights, y, forward_returns, config.meta_models, config, 'ensemble', None, None, None) if len(config.meta_models) > 0 else None
    
    return PipelineOutcome(directional_training_outcome, bet_sizing_outcomes, ensemble_outcome, ensemble_bet_sizing_outcome)
    
if __name__ == '__main__':
    run_pipeline(project_name='price-prediction', with_wandb = False, sweep = False, raw_config=get_default_ensemble_config())
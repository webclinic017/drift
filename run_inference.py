from data_loader import load_data
from data_loader.process import check_data
from reporting.saving import load_models

from run_pipeline import run_pipeline
from config.types import Config, RawConfig
from config.presets import get_dev_config, get_default_ensemble_config, get_lightweight_ensemble_config
from labeling.process import label_data
import pandas as pd

from training.directional_training import train_directional_models
from training.bet_sizing import bet_sizing_with_meta_models
from training.ensemble import ensemble_weights
from training.types import PipelineOutcome

def run_inference(preload_models:bool, fallback_raw_config: RawConfig):
    if preload_models:
        pipeline_outcome, config = load_models(None)
    else:
        pipeline_outcome, config = run_pipeline(project_name='price-prediction', with_wandb = False, sweep = False, raw_config=fallback_raw_config)
    
    config.mode = 'inference'
    __inference(config, pipeline_outcome)


def __inference(config: Config, pipeline_outcome: PipelineOutcome):
    
    # 1. Load data, check for validity and process data
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
    assert check_data(X, config) == True, "Data is not valid. Cancelling Inference." 

    events, X, y, forward_returns = label_data(config.event_filter, config.labeling, X, returns, forward_returns)

    inference_from: pd.Timestamp = X.index[len(X.index) - 1]

    # 2. Filter for significant events when we want to trade, and label data
    events, X, y, forward_returns = label_data(config.event_filter, config.labeling, X, returns, forward_returns)

    # 3. Train directional models
    directional_training_outcome = train_directional_models(X, y, forward_returns, config, config.directional_models, from_index = inference_from, preloaded_training_step = pipeline_outcome.directional_training)
    
    # 4. Run bet sizing on primary model's output
    bet_sizing_outcomes = [bet_sizing_with_meta_models(X, training_outcome.predictions, y, forward_returns, config.meta_models, config, 'meta', from_index = inference_from, transformations_over_time = preloaded_outcome.meta_transformations, preloaded_models = [b.model_over_time for b in preloaded_outcome.meta_training]) for training_outcome, preloaded_outcome in zip(directional_training_outcome.training, pipeline_outcome.bet_sizing)]

    # 4. Ensemble weights
    ensemble_outcome = ensemble_weights([o.weights for o in bet_sizing_outcomes], forward_returns, y, config.no_of_classes, config.mode == 'training')

    # 5. (Optional) Additional bet sizing on top of the ensembled weights
    ensemble_bet_sizing_outcome = bet_sizing_with_meta_models(X, ensemble_outcome.weights, y, forward_returns, config.meta_models, config, 'ensemble', from_index = inference_from, transformations_over_time = pipeline_outcome.secondary_bet_sizing.meta_transformations, preloaded_models= [b.model_over_time for b in pipeline_outcome.secondary_bet_sizing.meta_training]) if len(config.meta_models) > 0 else None
    
    return PipelineOutcome(directional_training_outcome, bet_sizing_outcomes, ensemble_outcome, ensemble_bet_sizing_outcome)


if __name__ == '__main__':
    run_inference(preload_models=True, fallback_raw_config=get_lightweight_ensemble_config())
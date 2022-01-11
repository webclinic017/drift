from training.inference import run_inference_pipeline
from models.saving import load_models

from run_pipeline import run_pipeline
from config.config import get_dev_config, get_default_ensemble_config, get_lightweight_ensemble_config

from typing import Callable


def run_inference(preload_models:bool, get_config:Callable):
    if preload_models:
        all_models_all_assets, data_config, training_config = load_models(None)
    else:
        all_models_all_assets, data_config, training_config, _, _, _ = run_pipeline(project_name='price-prediction', with_wandb = False, sweep = False, get_config=get_config)
    
    run_inference_pipeline(data_config, training_config, all_models_all_assets)
    

if __name__ == '__main__':
    run_inference(preload_models=False, get_config=get_lightweight_ensemble_config)
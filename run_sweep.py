from run_pipeline import run_pipeline
from config.config import get_default_ensemble_config

run_pipeline(project_name='price-prediction', with_wandb = True, sweep = True, get_config= get_default_ensemble_config)
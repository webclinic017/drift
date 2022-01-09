from run_pipeline import run_pipeline
from config.config import get_dev_config


run_pipeline(project_name='price-prediction', with_wandb = False, sweep = False, get_config=get_dev_config)

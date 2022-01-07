from run_pipeline import run_pipeline
from config.config import get_default_level_1_daily_config, get_default_level_2_daily_config, get_default_level_2_hourly_config


run_pipeline(project_name='price-prediction', with_wandb = False, sweep = False, get_config=get_default_level_1_daily_config)

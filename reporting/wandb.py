import pandas as pd
from config.types import RawConfig
from typing import Optional
from utils.helpers import weighted_average
from training.types import Stats


def launch_wandb(
    project_name: str, default_config: RawConfig, sweep: bool = False
) -> Optional[object]:
    from wandb_setup import get_wandb

    wandb = get_wandb()
    if wandb is None:
        raise Exception(
            "Wandb can not be initalized, the environment variable WANDB_API_KEY is missing (can also use .env file)"
        )

    elif sweep:
        wandb.init(project=project_name, config=vars(default_config))
        return wandb
    else:
        wandb.init(project=project_name, config=vars(default_config), reinit=True)
        return wandb


def override_config_with_wandb_values(
    wandb: Optional[object], raw_config: RawConfig
) -> RawConfig:
    if wandb is None:
        return raw_config

    wandb_config: dict = wandb.config

    config_dict = vars(raw_config)
    for k in config_dict:
        config_dict[k] = wandb_config[k]

    return RawConfig(**config_dict)


def send_report_to_wandb(stats: Stats, wandb: Optional[object]):
    if wandb is None:
        return

    run = wandb.run
    run.save()

    for key, value in stats.items():
        run.log({key: value})

    run.finish()

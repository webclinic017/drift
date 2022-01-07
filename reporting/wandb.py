import pandas as pd
from typing import Optional
from utils.helpers import weighted_average

def launch_wandb(project_name:str, default_config:dict, sweep:bool=False):
    from wandb_setup import get_wandb
    wandb = get_wandb()   
    
    if wandb is None: 
        return None
    elif sweep:
        wandb.init(project=project_name,  config = default_config)             
        return wandb
    else:
        wandb.init(project=project_name, config = default_config, reinit=True)
        return wandb
    
    
def register_config_with_wandb(wandb: Optional[object], model_config:dict, training_config:dict, data_config:dict):
    if wandb is None: return model_config, training_config, data_config

    config: dict = wandb.config
    
    for k in training_config:
        training_config[k] = config[k]
    for k in model_config:
        model_config[k] = config[k]
    for k in data_config:
        data_config[k] = config[k]

    return model_config, training_config, data_config

def send_report_to_wandb(results: pd.DataFrame, wandb:Optional[object], project_name: str, model_name: str):
    if wandb is None: return

    run = wandb.init(project=project_name, config={"model_type": model_name}, reinit=True)
    wandb.run.name = model_name+ "-" + wandb.run.id
    wandb.run.save()

    mean_results = weighted_average(results, 'no_of_samples')
    for key, value in mean_results.iteritems():
        run.log({"model_type": model_name, key: value })

    run.finish()


        
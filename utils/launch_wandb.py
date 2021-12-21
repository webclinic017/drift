

def launch_wandb(project_name:str, default_config:dict, sweep:bool=False):
    from wandb_setup import get_wandb
    wandb = get_wandb()   
    
    if type(wandb) == type(None): 
        return None
    elif sweep:
        wandb.init(project=project_name,  config = default_config)             
        return wandb
    else:
        wandb.init(project=project_name, config = default_config, reinit=True)
        return wandb
    
    
def seperate_configs(wandb, model_config:dict, training_config:dict, data_config:dict) -> tuple[dict,dict,dict]:
    config:dict = wandb.config

    if type(wandb) is not type(None):
        for k in training_config: training_config[k] = config[k]
        for k in model_config: model_config[k] = config[k]
        # for k in data_config: data_config[k] = config[k]

    return model_config, training_config, data_config



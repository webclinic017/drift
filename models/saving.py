import pickle
import datetime
from typing import Optional, Union
import os
import warnings



def save_models(all_models_for_all_assets: dict, data_config:dict, training_config:dict) -> None:
    all_models_for_all_assets['training_config'] = training_config
    all_models_for_all_assets['data_config'] = data_config
    
    date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    
    if not os.path.exists('output/models'):
        warnings.warn("No folder exists, creating one.")
        os.makedirs('output/models')
        
    pickle.dump( all_models_for_all_assets, open( "output/models/{}.p".format(date_string), "wb" ) )


def load_models(file_name:Union[str, None]) -> tuple[dict, dict, dict]:

    if file_name is None: 
        warnings.warn("No file name provided, will load latest models and configurations.")
        files_in_directory:list = os.listdir('output/models')
        
        assert len(files_in_directory) > 0, "No models found in output/models."
        file_name = sorted(files_in_directory)[-1]
    
    all_models_for_all_assets = pickle.load( open( "output/models/{}".format(file_name), "rb" ) )
    
    data_config = all_models_for_all_assets['data_config']
    training_config = all_models_for_all_assets['training_config']
    
    return all_models_for_all_assets, data_config, training_config


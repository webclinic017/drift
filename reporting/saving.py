import pickle
import datetime
from config.types import Config
from typing import Optional, Union
import os
import warnings
from reporting.types import Reporting



def save_models(all_models: Reporting.Asset, config: Config) -> None:
    dict_for_pickle = dict()
    dict_for_pickle['config'] = config
    dict_for_pickle['all_models'] = all_models
    
    date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    
    if not os.path.exists('output/models'):
        warnings.warn("No folder exists, creating one.")
        os.makedirs('output/models')
        
    pickle.dump( dict_for_pickle, open( "output/models/{}.p".format(date_string), "wb" ) )


def load_models(file_name:Union[str, None]) -> tuple[Reporting.Asset, Config]:

    if file_name is None: 
        warnings.warn("No file name provided, will load latest models and configurations.")
        files_in_directory:list = os.listdir('output/models')
        
        assert len(files_in_directory) > 0, "No models found in output/models."
        file_name = sorted(files_in_directory)[-1]
    
    packacked_dict = pickle.load( open( "output/models/{}".format(file_name), "rb" ) )
    
    config = packacked_dict.pop("config", None)    
    all_models = packacked_dict.pop("all_models", None)    
    
    return all_models, config


from .types import Config, RawConfig
from utils.helpers import flatten
from feature_extractors.feature_extractor_presets import presets as feature_extractor_presets
from models.model_map import get_model
from data_loader.collections import data_collections
from labeling.eventfilters_map import eventfilters_map
from labeling.labellers_map import labellers_map

def preprocess_config(raw_config: RawConfig) -> Config:
    config_dict = vars(raw_config)
    config_dict = __preprocess_model_config(config_dict)
    config_dict = __preprocess_feature_extractors_config(config_dict)
    config_dict = __preprocess_data_collections_config(config_dict)
    config_dict = __preprocess_event_filter_config(config_dict)
    config_dict = __preprocess_event_labeller_config(config_dict)

    config_dict['no_of_classes'] = 'two'
    config = Config(**config_dict)
    validate_config(config)
    return config

def __preprocess_feature_extractors_config(data_dict: dict) -> dict:
    data_dict = data_dict.copy()
    keys = ['own_features', 'other_features', 'exogenous_features']
    for key in keys:
        preset_names = data_dict[key]
        data_dict[key] = flatten([feature_extractor_presets[preset_name] for preset_name in preset_names])
    return data_dict

def __preprocess_model_config(model_config:dict) -> dict:
    model_config['directional_models'] = [get_model(model_name) for model_name in model_config['directional_models']]
    if len(model_config['meta_models']) > 0:
        model_config['meta_models'] = [get_model(model_name) for model_name in  model_config['meta_models']]

    return model_config

def __preprocess_data_collections_config(data_dict: dict) -> dict:
    keys = ['assets', 'other_assets', 'exogenous_data']
    for key in keys:
        preset_names = data_dict[key]
        data_dict[key] = flatten([data_collections[preset_name] for preset_name in preset_names])
    target_asset = next(iter([asset for asset in data_dict['assets'] if asset[1] == data_dict['target_asset']]), None)
    if target_asset is None: raise Exception('Target asset wasnt found in assets')
    data_dict['target_asset'] = target_asset
    return data_dict

def __preprocess_event_filter_config(data_dict: dict) -> dict:
    data_dict['event_filter'] = eventfilters_map[data_dict['event_filter']]
    return data_dict

def __preprocess_event_labeller_config(data_dict: dict) -> dict:
    data_dict['labeling'] = labellers_map[data_dict['labeling']]
    return data_dict
    

def validate_config(config: Config):
    # We need to make sure there's only one output from the pipeline
    # If meta model is there, we need more than one directional models to train
    if len(config.meta_models) > 1: assert len(config.directional_models) > 0
    # If there's no level-2 model, we need to have only one level-1 model
    if len(config.meta_models) == 0: assert len(config.directional_models) == 1



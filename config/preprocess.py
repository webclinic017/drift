
from utils.helpers import flatten
from feature_extractors.feature_extractor_presets import presets as feature_extractor_presets
from models.model_map import get_model_map
from data_loader.collections import data_collections

def preprocess_config(model_config:dict, training_config:dict, data_config:dict) -> tuple[dict, dict, dict]:
    model_config = __preprocess_model_config(model_config, data_config['method'])
    data_config = __preprocess_feature_extractors_config(data_config)
    data_config = __preprocess_data_collections_config(data_config)

    validate_config(model_config, training_config, data_config)
    return model_config, training_config, data_config

def __preprocess_feature_extractors_config(data_dict: dict) -> dict:
    data_dict = data_dict.copy()
    keys = ['own_features', 'other_features', 'exogenous_features']
    for key in keys:
        preset_names = data_dict[key]
        data_dict[key] = flatten([feature_extractor_presets[preset_name] for preset_name in preset_names])
    return data_dict

def __preprocess_model_config(model_config:dict, method:str) -> dict:
    model_map = get_model_map(model_config)
    model_config['primary_models'] = [(model_name, model_map['primary_models'][model_name]) for model_name in model_config['primary_models']]
    if len(model_config['meta_labeling_models']) > 0:
        model_config['meta_labeling_models'] = [(model_name, model_map['primary_models'][model_name]) for model_name in  model_config['meta_labeling_models']]
    if model_config['ensemble_model'] is not None:
        model_config['ensemble_model'] = (model_config['ensemble_model'], model_map['ensemble_models'][model_config['ensemble_model']])

    return model_config

def __preprocess_data_collections_config(data_dict: dict) -> dict:
    data_dict = data_dict.copy()
    keys = ['assets', 'other_assets', 'exogenous_data']
    for key in keys:
        preset_names = data_dict[key]
        data_dict[key] = flatten([data_collections[preset_name] for preset_name in preset_names])
    target_asset = next(iter([asset for asset in data_dict['assets'] if asset[1] == data_dict['target_asset']]), None)
    if target_asset is None: raise Exception('Target asset wasnt found in assets')
    data_dict['target_asset'] = target_asset
    return data_dict


def validate_config(model_config:dict, training_config:dict, data_config:dict):
    # We need to make sure there's only one output from the pipeline
    # If level-2 model is there, we need more than one level-1 models to train
    if len(model_config["meta_labeling_models"]) > 1: assert len(model_config["primary_models"]) > 0
    # If there's no level-2 model, we need to have only one level-1 model
    if len(model_config["meta_labeling_models"]) == 0: assert len(model_config["primary_models"]) == 1

def get_model_name(model_config:dict) -> str:
    if len(model_config["meta_labeling_models"]) > 0:
        return model_config["meta_labeling_models"][0][0]
    elif len(model_config["primary_models"]) == 1:
        return model_config["primary_models"][0][0]
    else:
        raise Exception("No model name found")


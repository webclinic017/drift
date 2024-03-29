from .types import Config, RawConfig
from utils.helpers import flatten
from feature_extractors.feature_extractor_presets import (
    presets as feature_extractor_presets,
)
from models.model_map import get_model
from data_loader.collections import data_collections
from labeling.eventfilters_map import eventfilters_map
from labeling.labellers_map import labellers_map
from models.sklearn import SKLearnModel
from sklearn.ensemble import VotingClassifier, StackingClassifier
from transformations.retrieve import get_pca, get_rfe, get_scaler
from copy import deepcopy
from models.base import Model
from typing import Literal


def preprocess_config(raw_config: RawConfig) -> Config:
    config_dict = vars(deepcopy(raw_config))
    config_dict = __preprocess_model_config(config_dict)
    config_dict = __preprocess_feature_extractors_config(config_dict)
    config_dict = __preprocess_data_collections_config(config_dict)
    config_dict = __preprocess_event_filter_config(config_dict)
    config_dict = __preprocess_event_labeller_config(config_dict)
    config_dict = __preprocess_transformations_config(config_dict)

    config_dict["no_of_classes"] = "two"
    config_dict["mode"] = "training"
    config = Config(**config_dict)
    return config


def __preprocess_feature_extractors_config(data_dict: dict) -> dict:
    data_dict = data_dict.copy()
    keys = ["own_features", "other_features", "exogenous_features"]
    for key in keys:
        preset_names = data_dict[key]
        data_dict[key] = flatten(
            [feature_extractor_presets[preset_name] for preset_name in preset_names]
        )
    return data_dict


def __preprocess_model_config(model_config: dict) -> dict:
    def get_ensemble_model(
        estimators: list[Model], method: Literal["voting_soft", "stacking"]
    ) -> Model:

        if method == "voting_soft":
            return SKLearnModel(
                VotingClassifier(
                    [(m.name, m) for m in directional_models],
                    voting="soft",
                )
            )
        elif method == "stacking":
            return SKLearnModel(
                StackingClassifier(
                    [(m.name, m) for m in estimators],
                    final_estimator=estimators[0],
                    cv=5,
                )
            )
        else:
            raise Exception(f"Unknown ensembling method: {method}")

    directional_models = [
        get_model(model_name) for model_name in model_config["directional_models"]
    ]
    model_config.pop("directional_models")

    if len(directional_models) > 1:
        model_config["directional_model"] = get_ensemble_model(
            directional_models, method=model_config["ensembling_method"]
        )
    else:
        model_config["directional_model"] = directional_models[0]

    meta_models = [get_model(model_name) for model_name in model_config["meta_models"]]
    if len(model_config["meta_models"]) > 1:
        model_config["meta_model"] = get_ensemble_model(
            meta_models, method=model_config["ensembling_method"]
        )
    else:
        model_config["meta_model"] = meta_models[0]

    model_config.pop("meta_models")
    model_config.pop("ensembling_method")

    return model_config


def __preprocess_data_collections_config(data_dict: dict) -> dict:
    keys = ["assets", "other_assets", "exogenous_data"]
    for key in keys:
        preset_names = data_dict[key]
        data_dict[key] = flatten(
            [data_collections[preset_name] for preset_name in preset_names]
        )
    target_asset = next(
        iter(
            [
                asset
                for asset in data_dict["assets"]
                if asset.file_name == data_dict["target_asset"]
            ]
        ),
        None,
    )
    if target_asset is None:
        raise Exception("Target asset wasnt found in assets")
    data_dict["target_asset"] = target_asset
    return data_dict


def __preprocess_event_filter_config(config_dict: dict) -> dict:
    config_dict["event_filter"] = eventfilters_map[config_dict["event_filter"]](
        config_dict["event_filter_multiplier"]
    )
    config_dict.pop("event_filter_multiplier")
    return config_dict


def __preprocess_event_labeller_config(config_dict: dict) -> dict:
    config_dict["labeling"] = labellers_map[config_dict["labeling"]](
        config_dict["forecasting_horizon"]
    )
    return config_dict


def __preprocess_transformations_config(config_dict: dict) -> dict:
    transformations = [
        get_scaler(config_dict["scaler"]),
        get_pca(
            config_dict["dimensionality_reduction_ratio"],
            config_dict["initial_window_size"],
        ),
        get_rfe(config_dict["n_features_to_select"]),
    ]
    transformations = [x for x in transformations if x is not None]
    config_dict["transformations"] = transformations
    config_dict.pop("scaler")
    config_dict.pop("dimensionality_reduction_ratio")
    config_dict.pop("n_features_to_select")

    return config_dict

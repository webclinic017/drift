from data_loader.types import DataCollection


def hash_data_config(data_config: dict) -> str:
    def hash_data_collection(data_collection: DataCollection) -> str:
        return "".join([a.path + a.file_name for a in data_collection])

    def hash_feature_extractors(feature_extractos) -> str:
        return "".join([f[0] for f in feature_extractos])

    def to_str(x):
        return "".join([str(i) for i in x])

    return "_".join(
        to_str(
            [
                hash_data_collection(data_config["assets"]),
                hash_data_collection(data_config["other_assets"]),
                hash_data_collection(data_config["exogenous_data"]),
                data_config["target_asset"].path
                + data_config["target_asset"].file_name,
                data_config["load_non_target_asset"],
                hash_feature_extractors(data_config["own_features"]),
                hash_feature_extractors(data_config["other_features"]),
                hash_feature_extractors(data_config["exogenous_features"]),
            ]
        )
    )


from utils.types import DataCollection

def hash_data_config(data_config: dict) -> str:
    
    def hash_data_collection(data_collection: DataCollection) -> str: return ''.join([a[0] + a[1] for a in data_collection])
    def hash_feature_extractors(feature_extractos) -> str: return ''.join([f[0] for f in feature_extractos])
    def to_str(x): return ''.join([str(i) for i in x])
    return '_'.join(to_str([
        hash_data_collection(data_config['assets']),
        hash_data_collection(data_config['other_assets']),
        hash_data_collection(data_config['exogenous_data']),
        data_config['target_asset'][0] + data_config['target_asset'][1],
        data_config['load_non_target_asset'],
        data_config['log_returns'],
        data_config['forecasting_horizon'],
        hash_feature_extractors(data_config['own_features']),
        hash_feature_extractors(data_config['other_features']),
        hash_feature_extractors(data_config['exogenous_features']),
        data_config['index_column'],
        data_config['no_of_classes'],
        data_config['narrow_format']
    ]))

from utils.evaluate import discretize_threeway_threshold, evaluate_predictions
from utils.helpers import random_string, equal_except_nan, drop_until_first_valid_index
from training.training import run_single_asset_trainig
from feature_selection.feature_selection import select_features
import pandas as pd
from models.model_map import default_feature_selector_regression, default_feature_selector_classification


def run_meta_labeling_training(
                            target_asset: str,
                            X_pca: pd.DataFrame,
                            input_predictions: pd.Series,
                            y: pd.Series,
                            target_returns: pd.Series,
                            data_config: dict,
                            model_config: dict,
                            training_config: dict
                        ) -> tuple[pd.Series, pd.Series, pd.DataFrame, dict]:

    discretize = discretize_threeway_threshold(0.33)
    discretized_predictions = input_predictions.apply(discretize)
    meta_y: pd.Series = pd.concat([discretized_predictions, y], axis=1).apply(equal_except_nan, axis = 1)

    print("Feature Selection started")
    backup_model = default_feature_selector_regression if data_config['method'] == 'regression' else default_feature_selector_classification
    meta_feature_selection_input_X, meta_feature_selection_input_y = drop_until_first_valid_index(X_pca, meta_y)
    feature_selection_output = select_features(X = meta_feature_selection_input_X, y = meta_feature_selection_input_y, model = model_config['level_1_models'][0][1], n_features_to_select = training_config['n_features_to_select'], backup_model = backup_model, scaling = training_config['scaler'], data_config_hash = random_string(10))
    meta_selected_features_X = X_pca[feature_selection_output.columns]

    meta_X = pd.concat([meta_selected_features_X, input_predictions, discretized_predictions], axis = 1)

    _, meta_preds, meta_probabilities, all_models_single_asset = run_single_asset_trainig(
        ticker_to_predict = "prediction_correct",
        original_X = meta_X,
        X = meta_X,
        y = meta_y,
        target_returns = target_returns,
        models = [model_config['level_2_model']],
        method = data_config['method'],
        expanding_window = training_config['expanding_window_level2'],
        sliding_window_size = training_config['sliding_window_size_level2'],
        retrain_every = training_config['retrain_every'],
        scaler = training_config['scaler'],
        no_of_classes = 'two',
        level = 2
    )
    bet_size = meta_probabilities.iloc[:,1]
    avg_predictions_with_sizing = input_predictions * bet_size
    avg_predictions_with_sizing.rename("model_" + target_asset + "_meta_lvl" + str(2), inplace=True)

    meta_result = evaluate_predictions(
        model_name = "Meta",
        target_returns = target_returns,
        y_pred = avg_predictions_with_sizing,
        y_true = y,
        method = 'classification',
        no_of_classes = data_config['no_of_classes'],
        discretize=False
    )
    meta_result.rename("model_" + target_asset + "_meta_lvl" + str(2), inplace=True)

    return meta_result, avg_predictions_with_sizing, meta_probabilities, all_models_single_asset
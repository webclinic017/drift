import pandas as pd
from operator import itemgetter

from training.primary_model import train_primary_model
from training.meta_labeling import train_meta_labeling_model

from reporting.types import Reporting
from typing import Union


def primary_step(
                X: pd.DataFrame,
                y:pd.Series,
                asset:list,
                target_returns:pd.Series,
                configs: dict,
                reporting: Reporting,
                preloaded_training_step: Union[Reporting.Training_Step, None] = None
                ) -> tuple[Reporting.Training_Step, pd.DataFrame]:
    training_step = Reporting.Training_Step(level='primary')
    model_config, training_config, data_config = itemgetter('model_config', 'training_config', 'data_config')(configs)

    # 3. Train Primary models
    current_result, current_predictions, current_probabilities, all_models_for_single_asset = train_primary_model(
        ticker_to_predict = asset[1],
        X = X,
        y = y,
        target_returns = target_returns,
        models = model_config['primary_models'],
        method = data_config['method'],
        expanding_window = training_config['expanding_window_primary'],
        sliding_window_size = training_config['sliding_window_size_primary'],
        retrain_every =  training_config['retrain_every'],
        scaler =  training_config['scaler'],
        no_of_classes = data_config['no_of_classes'],
        level = 'primary',
        print_results= True,
        preloaded_models = preloaded_training_step.convert_step_to_tuple('base') if preloaded_training_step is not None else None
    )
    
    training_step.base = all_models_for_single_asset
    
    # 4. Train a Meta-Labeling model for each Primary model and replace their predictions with the meta-labeling predictions
    if training_config['primary_models_meta_labeling'] == True:
        for model_name in current_result.columns:
            primary_model_predictions = current_predictions[model_name]
            primary_meta_result, primary_meta_preds, primary_meta_probabilities, meta_labeling_models = train_meta_labeling_model(
                target_asset=asset[1],
                X = X,
                input_predictions= primary_model_predictions,
                y = y,
                target_returns = target_returns,
                models = model_config['meta_labeling_models'],
                data_config= data_config,
                model_config= model_config,
                training_config= training_config,
                model_suffix = 'meta',
                preloaded_models =preloaded_training_step.convert_step_to_tuple('metalabeling') if preloaded_training_step is not None else None
            )
            current_result[model_name] = primary_meta_result
            current_predictions[model_name] = primary_meta_preds

            training_step.metalabeling.append(meta_labeling_models)
        
    reporting.results = pd.concat([reporting.results, current_result], axis=1)
    # With static models, because of the lag in the indicator, the first prediction is NA, so we fill it with zero.
    reporting.all_predictions = pd.concat([reporting.all_predictions, current_predictions], axis=1).fillna(0.)
    reporting.all_probabilities = pd.concat([reporting.all_probabilities, current_probabilities], axis=1).fillna(0.)    
    
    return training_step, current_predictions


def secondary_step(
                X:pd.DataFrame,
                y:pd.Series,
                current_predictions:pd.DataFrame,
                asset:list,
                target_returns:pd.Series,
                configs: dict,
                reporting: Reporting
                ) -> Reporting.Training_Step:
    training_step = Reporting.Training_Step(level='secondary')
    model_config, training_config, data_config = itemgetter('model_config', 'training_config', 'data_config')(configs)
    
    # 5. Ensemble primary model predictions (If Ensemble model is present)
    if model_config['ensemble_model'] is not None:
        ensemble_result, ensemble_predictions, _, ensemble_models_one_asset = train_primary_model(
            ticker_to_predict = asset[1],
            X = current_predictions,
            y = y,
            target_returns = target_returns,
            models = [model_config['ensemble_model']],
            method = data_config['method'],
            expanding_window = False,
            sliding_window_size = 1,
            retrain_every = training_config['retrain_every'],
            scaler = training_config['scaler'],
            no_of_classes = data_config['no_of_classes'],
            level = 'ensemble',
            print_results= True,
        )
        ensemble_result, ensemble_predictions = ensemble_result.iloc[:,0], ensemble_predictions.iloc[:,0]
        
        training_step.base = ensemble_models_one_asset
        
        reporting.results = pd.concat([reporting.results, ensemble_result], axis=1)
        reporting.all_predictions = pd.concat([reporting.all_predictions, ensemble_predictions], axis=1)
    
        
        if len(model_config['meta_labeling_models']) > 0: 

            # 3. Train a Meta-labeling model on the averaged level-1 model predictions
            ensemble_meta_result, ensemble_meta_predictions, ensemble_meta_probabilities, ensemble_meta_labeling_models = train_meta_labeling_model(
                target_asset=asset[1],
                X = X,
                input_predictions= ensemble_predictions,
                y = y,
                target_returns = target_returns,
                models = model_config['meta_labeling_models'],
                data_config= data_config,
                model_config= model_config,
                training_config= training_config,
                model_suffix = 'ensemble'
            )
            
            training_step.metalabeling.append(ensemble_meta_labeling_models)
            
            
            reporting.results = pd.concat([reporting.results, ensemble_meta_result], axis=1)
            reporting.all_predictions = pd.concat([reporting.all_predictions, ensemble_meta_predictions], axis=1)
            reporting.all_probabilities = pd.concat([reporting.all_probabilities, ensemble_meta_probabilities], axis=1).fillna(0.)
            
    return training_step

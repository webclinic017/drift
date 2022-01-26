from numpy import DataSource
import pandas as pd
from operator import itemgetter

from training.primary_model import train_primary_model
from training.meta_labeling import train_meta_labeling_model

from reporting.types import Reporting
from typing import Union, Optional
from config.types import Config 


def primary_step(
                X: pd.DataFrame,
                y: pd.Series,
                forward_returns: pd.Series,
                config: Config,
                reporting: Reporting,
                from_index: Optional[pd.Timestamp],
                preloaded_training_step: Optional[Reporting.Training_Step] = None,
                ) -> tuple[Reporting.Training_Step, pd.DataFrame]:
    training_step = Reporting.Training_Step(level='primary')

    # 3. Train Primary models
    current_result, current_predictions, current_probabilities, all_models_for_single_asset = train_primary_model(
        ticker_to_predict = config.target_asset[1],
        X = X,
        y = y,
        forward_returns = forward_returns,
        models = config.primary_models,
        expanding_window = config.expanding_window_base,
        sliding_window_size = config.sliding_window_size_base,
        retrain_every =  config.retrain_every,
        from_index = from_index,
        scaler =  config.scaler,
        no_of_classes = config.no_of_classes,
        level = 'primary',
        print_results= True,
        preloaded_models = preloaded_training_step.get_base() if preloaded_training_step is not None else None
    )
    
    training_step.base = all_models_for_single_asset
    
    # 4. Train a Meta-Labeling model for each Primary model and replace their predictions with the meta-labeling predictions
    if config.primary_models_meta_labeling == True:
        for model_name in current_result.columns:
            primary_model_predictions = current_predictions[model_name]
            primary_meta_result, primary_meta_preds, primary_meta_probabilities, meta_labeling_models = train_meta_labeling_model(
                target_asset = config.target_asset[1],
                X = X,
                input_predictions= primary_model_predictions,
                y = y,
                forward_returns = forward_returns,
                model_suffix = 'meta',
                models = config.meta_labeling_models,
                config = config,
                from_index = from_index,
                preloaded_models = preloaded_training_step.get_metalabeling()[model_name] if preloaded_training_step is not None else None
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
                forward_returns:pd.Series,
                config: Config,
                reporting: Reporting,
                from_index: Optional[pd.Timestamp],
                preloaded_training_step: Optional[Reporting.Training_Step] = None,
                ) -> Reporting.Training_Step:
    training_step = Reporting.Training_Step(level='secondary')
    
    # 5. Ensemble primary model predictions (If Ensemble model is present)
    if config.ensemble_model is not None:
        ensemble_result, ensemble_predictions, _, ensemble_models_one_asset = train_primary_model(
            ticker_to_predict = config.target_asset[1],
            X = current_predictions,
            y = y,
            forward_returns = forward_returns,
            models = [config.ensemble_model],
            expanding_window = False,
            sliding_window_size = 1,
            retrain_every = config.retrain_every,
            from_index = from_index,
            scaler = config.scaler,
            no_of_classes = config.no_of_classes,
            level = 'ensemble',
            print_results= True,
            preloaded_models = preloaded_training_step.get_base() if preloaded_training_step is not None else None
        )
        ensemble_result, ensemble_predictions = ensemble_result.iloc[:,0], ensemble_predictions.iloc[:,0]
        
        training_step.base = ensemble_models_one_asset
        
        reporting.results = pd.concat([reporting.results, ensemble_result], axis=1)
        reporting.all_predictions = pd.concat([reporting.all_predictions, ensemble_predictions], axis=1)
    
        
        if len(config.meta_labeling_models) > 0: 

            # 3. Train a Meta-labeling model on the averaged level-1 model predictions
            ensemble_meta_result, ensemble_meta_predictions, ensemble_meta_probabilities, ensemble_meta_labeling_models = train_meta_labeling_model(
                target_asset = config.target_asset[1],
                X = X,
                input_predictions= ensemble_predictions,
                y = y,
                forward_returns = forward_returns,
                models = config.meta_labeling_models,
                config = config,
                model_suffix = 'ensemble',
                from_index = from_index,
                preloaded_models = preloaded_training_step.get_metalabeling()[ensemble_predictions.name] if preloaded_training_step is not None else None
            )
            
            training_step.metalabeling.append(ensemble_meta_labeling_models)
            
            
            reporting.results = pd.concat([reporting.results, ensemble_meta_result], axis=1)
            reporting.all_predictions = pd.concat([reporting.all_predictions, ensemble_meta_predictions], axis=1)
            reporting.all_probabilities = pd.concat([reporting.all_probabilities, ensemble_meta_probabilities], axis=1).fillna(0.)
            
    return training_step

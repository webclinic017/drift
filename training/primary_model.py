import pandas as pd
from typing import Literal, Optional, Union
from training.walk_forward import walk_forward_train, walk_forward_inference
from utils.evaluate import evaluate_predictions
from models.base import Model
from utils.scaler import get_scaler
from utils.types import ScalerTypes
from reporting.types import Reporting
from transformations.rfe import RFETransformation
from transformations.pca import PCATransformation

def train_primary_model(
                    ticker_to_predict: str,
                    X: pd.DataFrame,
                    y: pd.Series,
                    target_returns: pd.Series,
                    models: list[tuple[str, Model]],
                    expanding_window: bool,
                    sliding_window_size: int,
                    retrain_every: int,
                    from_index: Optional[int],
                    scaler: ScalerTypes,
                    no_of_classes: Literal['two', 'three-balanced', 'three-imbalanced'],
                    level: str,
                    print_results: bool,
                    preloaded_models: Optional[list[tuple[str, pd.Series, list[pd.Series]]]] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Reporting.Single_Model]]:

    results = pd.DataFrame()
    predictions = pd.DataFrame(index=y.index)
    probabilities = pd.DataFrame(index=y.index)
    all_models_single_asset: list[Reporting.Single_Model] = []

    unified_models: list[tuple[str, pd.Series, list[pd.Series]]] = []
    if preloaded_models is not None:
        unified_models = preloaded_models
    
    transformations_over_time = None
    
    
    if preloaded_models is None:
        train_unified_models=[]
        for model_name, model in models:
            model_over_time, transformations_over_time = walk_forward_train(
                model_name=model_name,
                model = model,
                X = X,
                y = y,
                target_returns = target_returns,
                expanding_window = expanding_window,
                window_size = sliding_window_size,
                retrain_every = retrain_every,
                from_index = from_index,
                transformations= [
                    get_scaler(scaler),
                    PCATransformation(ratio_components_to_keep=0.5, sliding_window_size=sliding_window_size),
                    RFETransformation(n_feature_to_select=40, model=model)
                ],
                preloaded_transformations=transformations_over_time,
            )
            train_unified_models.append((model_name, model_over_time, transformations_over_time))
        unified_models = train_unified_models
        
    for model_tuples in unified_models:
        model_name, model_over_time, transformations_over_time = model_tuples[0], model_tuples[1], model_tuples[2]
        preds, probs = walk_forward_inference(
            model_name = model_name,
            model_over_time= pd.Series(model_over_time),
            transformations_over_time = transformations_over_time,
            X = X,
            expanding_window = expanding_window,
            window_size = sliding_window_size,
            from_index = from_index,
        )
        
        assert len(preds) == len(y)
        result = evaluate_predictions(
            model_name = model_name,
            target_returns = target_returns,
            y_pred = preds,
            y_true = y,
            no_of_classes=no_of_classes,
            print_results = print_results,
            discretize=True
        )
        levelname=("_" + level) if level=='metalabeling' else ""
        if preloaded_models is None:
            column_name = "model_" + model_name + "_" + ticker_to_predict + levelname 
        else:
            column_name = model_name
        results[column_name] = result
        

        all_models_single_asset.append(Reporting.Single_Model(model_name=column_name, model_over_time=model_over_time, transformations_over_time=transformations_over_time))
  
        # column names for model outputs should be different, so we can differentiate between original data and model predictions later, where necessary
        predictions[column_name] = preds
        probs_column_name = "probs_" + ticker_to_predict + "_" + model_name + "_" + level
        probs.columns = [probs_column_name + "_" + c for c in probs.columns]
        probabilities = pd.concat([probabilities, probs], axis=1)
    


    return results, predictions, probabilities, all_models_single_asset
import pandas as pd
from typing import Literal
from training.walk_forward import walk_forward_train_test
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from utils.evaluate import evaluate_predictions
from models.base import Model

def __get_scaler(type: Literal['normalize', 'minmax', 'standardize', 'none']):
    if type == 'normalize':
        return Normalizer()
    elif type == 'minmax':
        return MinMaxScaler(feature_range= (-1, 1))
    elif type == 'standardize':
        return StandardScaler()
    else:
        return None

def run_single_asset_trainig(
                    ticker_to_predict: str,
                    X: pd.DataFrame,
                    y: pd.Series,
                    target_returns: pd.Series,
                    models: list[tuple[str, Model]],
                    method: Literal['regression', 'classification'],
                    sliding_window_size: int,
                    retrain_every: int,
                    scaler: Literal['normalize', 'minmax', 'standardize', 'none'],
                    wandb,
                    project_name:str,
                    sweep:bool
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:


    scaler = __get_scaler(scaler)

    results = pd.DataFrame()
    predictions = pd.DataFrame()
    
    wandb_active = type(wandb) is not type(None)

    for model_name, model in models:
        model_over_time, preds = walk_forward_train_test(
            model_name=model_name,
            model = model,
            X = X,
            y = y,
            target_returns = target_returns,
            window_size = sliding_window_size,
            retrain_every = retrain_every,
            scaler = scaler
        )
        assert len(preds) == len(y)
        result = evaluate_predictions(
            model_name = model_name,
            target_returns = target_returns,
            y_pred = preds,
            method = method,
        )
        column_name = ticker_to_predict + "_" + model_name
        results[column_name] = result
        # column names for model outputs should be different, so we can differentiate between original data and model predictions later, where necessary
        predictions["model_" + column_name] = preds
        
        if wandb_active and not sweep:
                run = wandb.init(project=project_name, config={"model_type": model_name, "ticker": ticker_to_predict}, reinit=True)
                wandb.run.name = ticker_to_predict + "-" + model_name+ "-" + wandb.run.id
                wandb.run.save()
            
                for rownum,(indx,val) in enumerate(result.iteritems()):
                    run.log({"model_type": model_name, indx:val })
                    
                run.finish()
    
    if wandb_active and sweep:
        mean_results = results.mean()
         
        wandb.log({"model_type": 'avarage_model', 'results':results })
        for rownum,(indx,val) in enumerate(mean_results.iteritems()):
            wandb.log({"model_type": 'avarage_model', indx:val })

    return results, predictions
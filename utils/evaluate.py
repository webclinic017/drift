from typing import Literal
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, r2_score, classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

def format_data_for_backtest(aggregated_data: pd.DataFrame, returns_col: str, only_test_data: pd.DataFrame, preds) -> pd.DataFrame:
    backtest_data = aggregated_data.iloc[-only_test_data.shape[0]:].copy()[returns_col]
    assert backtest_data.shape[0] == only_test_data.shape[0]
    backtest_data = backtest_data.reset_index(drop=True)    
    return pd.concat([backtest_data, pd.Series(preds)], axis='columns')


def __preprocess(y_true: pd.Series, y_pred: pd.Series, method: Literal['classification', 'regression']):
    y_pred.name = 'y_pred'
    y_true.name = 'y_true'
    df = pd.concat([y_pred, y_true],axis=1).dropna()

    if method == 'regression':
        df['sign_pred'] = df.y_pred.apply(np.sign)
    else:
        df['sign_pred'] = df.y_pred.apply(lambda x: 1 if x>0 else -1)
    df['sign_true'] = df.y_true.apply(np.sign)
    df['is_correct'] = 0
    df.loc[df.sign_pred * df.sign_true > 0 ,'is_correct'] = 1 # only registers 1 when prediction was made AND it was correct
    df['is_incorrect'] = 0
    df.loc[df.sign_pred * df.sign_true < 0,'is_incorrect'] = 1 # only registers 1 when prediction was made AND it was wrong
    df['is_predicted'] = df.is_correct + df.is_incorrect
    df['result'] = df.sign_pred * df.y_true 
    return df

def evaluate_predictions(model_name: str, y_true: pd.Series, y_pred: pd.Series, sliding_window_size: int, method: Literal['classification', 'regression']):
    evaluate_from = sliding_window_size+1
    y_true = pd.Series(y_true[evaluate_from:])
    y_pred = pd.Series(y_pred[evaluate_from:])

    df = __preprocess(y_true, y_pred, method)
    
    scorecard = pd.Series()
    if method == 'regression':
        scorecard.loc['RSQ'] = r2_score(df.y_true,df.y_pred)
        scorecard.loc['MAE'] = mean_absolute_error(df.y_true,df.y_pred)
    elif method == 'classification':
        scorecard.loc['RSQ'] = 0.
        scorecard.loc['MAE Matrix'] = 0.
    scorecard.loc['directional_accuracy'] = df.is_correct.sum()*1. / (df.is_predicted.sum()*1.)*100
    scorecard.loc['edge'] = df.result.mean()
    scorecard.loc['noise'] = df.y_pred.diff().abs().mean()
    scorecard.loc['edge_to_noise'] = scorecard.loc['edge'] / scorecard.loc['noise']
    if method == 'regression':
        scorecard.loc['edge_to_mae'] = scorecard.loc['edge'] / scorecard.loc['MAE']
    elif method == 'classification':
        scorecard.loc['edge_to_mae'] = 0.

    # TODO: add confusion matrix, f1 score, precision, recall
    print("Model name: ", model_name)
    print(scorecard)
    return scorecard  


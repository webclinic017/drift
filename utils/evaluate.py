from typing import Literal
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score, f1_score, precision_score, recall_score
from quantstats.stats import expected_return, sharpe, skew, sortino
import pandas as pd
import numpy as np

def backtest(returns: pd.Series, signal: pd.Series, transaction_cost = 0.02) -> pd.Series:
    delta_pos = signal.diff(1).abs().fillna(0.)
    costs = transaction_cost * delta_pos
    return (signal * returns) - costs


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
    df['result'] = backtest(df.y_true, df.sign_pred)

    return df

def evaluate_predictions(model_name: str, y_true: pd.Series, y_pred: pd.Series, sliding_window_size: int, method: Literal['classification', 'regression']):
    # ignore the predictions until we see a non-zero returns (and definitely skip the first sliding_window_size)
    first_nonzero_return = np.where(y_true != 0)[0][0]
    evaluate_from = first_nonzero_return + sliding_window_size + 1
    
    y_true = pd.Series(y_true[evaluate_from:])
    # if there are lots of zeros in the ground truth returns, probably something is wrong, but we can tolerate a couple of days of missing data.
    is_zero = y_true[y_true == 0]
    assert len(is_zero) < 5
    # we can't deal with 0 returns, so we'll just remap the few examples to 1
    y_true = y_true.apply(lambda x: 1 if x == 0 else x)
    y_pred = pd.Series(y_pred[evaluate_from:])

    df = __preprocess(y_true, y_pred, method)
    
    scorecard = pd.Series()
    if method == 'regression':
        scorecard.loc['RSQ'] = r2_score(df.y_true, df.y_pred)
        scorecard.loc['MAE'] = mean_absolute_error(df.y_true, df.y_pred)
    elif method == 'classification':
        scorecard.loc['RSQ'] = 0.
        scorecard.loc['MAE Matrix'] = 0.
    sign_true = df.sign_true.astype(int)
    sign_pred = df.sign_pred.astype(int)
    
    scorecard.loc['sharpe'] = sharpe(df.result)
    scorecard.loc['sortino'] = sortino(df.result)
    scorecard.loc['skew'] = skew(df.result)

    scorecard.loc['accuracy'] = accuracy_score(sign_true, sign_pred) * 100
    scorecard.loc['recall'] = recall_score(sign_true, sign_pred, labels = [1, -1])
    scorecard.loc['precision'] = precision_score(sign_true, sign_pred, labels = [1, -1])
    scorecard.loc['f1_score'] = f1_score(sign_true, sign_pred, labels = [1, -1])
    scorecard.loc['edge'] = df.result.mean()
    scorecard.loc['noise'] = df.y_pred.diff().abs().mean()
    scorecard.loc['edge_to_noise'] = scorecard.loc['edge'] / scorecard.loc['noise']


    if method == 'regression':
        scorecard.loc['edge_to_mae'] = scorecard.loc['edge'] / scorecard.loc['MAE']
    elif method == 'classification':
        scorecard.loc['edge_to_mae'] = 0.

    scorecard = scorecard.round(3)
    print("Model name: ", model_name)
    print(scorecard)
    return scorecard  


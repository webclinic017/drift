from typing import Literal
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score, f1_score, precision_score, recall_score
from quantstats.stats import expected_return, sharpe, skew, sortino
from utils.helpers import get_first_valid_return_index
import pandas as pd
import numpy as np

def backtest(returns: pd.Series, signal: pd.Series, transaction_cost = 0.00) -> pd.Series:
    delta_pos = signal.diff(1).abs().fillna(0.)
    costs = transaction_cost * delta_pos
    return (signal * returns) - costs


def __preprocess(target_returns: pd.Series, y_pred: pd.Series, method: Literal['classification', 'regression']) -> pd.DataFrame:
    y_pred.name = 'y_pred'
    target_returns.name = 'target_returns'
    df = pd.concat([y_pred, target_returns],axis=1).dropna()

    df['sign_pred'] = df.y_pred.apply(lambda x: 1 if x>0 else -1)
    def sign_true(x):
        if x > 0:
            return 1
        else:
            return -1
    df['sign_true'] = df.target_returns.apply(sign_true)
    df['is_correct'] = 0
    df.loc[df.sign_pred * df.sign_true > 0 ,'is_correct'] = 1 # only registers 1 when prediction was made AND it was correct
    df['is_incorrect'] = 0
    df.loc[df.sign_pred * df.sign_true < 0,'is_incorrect'] = 1 # only registers 1 when prediction was made AND it was wrong
    df['is_predicted'] = df.is_correct + df.is_incorrect
    df['result'] = backtest(df.target_returns, df.sign_pred)

    return df

def evaluate_predictions(
                        model_name: str,
                        target_returns: pd.Series,
                        y_pred: pd.Series,
                        method: Literal['classification', 'regression']
                        ) -> pd.Series:
    # ignore the predictions until we see a non-zero returns (and definitely skip the first sliding_window_size)
    first_nonzero_return = max(get_first_valid_return_index(target_returns), get_first_valid_return_index(y_pred))
    evaluate_from = first_nonzero_return + 1
    
    target_returns = pd.Series(target_returns[evaluate_from:])
    if method == 'regression':
        # if there are lots of zeros in the ground truth returns, probably something is wrong, but we can tolerate a couple of days of missing data.
        is_zero = target_returns[target_returns == 0]
        assert len(is_zero) < 15

    y_pred = pd.Series(y_pred[evaluate_from:])

    df = __preprocess(target_returns, y_pred, method)
    
    scorecard = pd.Series()
    if method == 'regression':
        scorecard.loc['RSQ'] = r2_score(df.target_returns, df.y_pred)
        scorecard.loc['MAE'] = mean_absolute_error(df.target_returns, df.y_pred)
    elif method == 'classification':
        scorecard.loc['RSQ'] = 0.
        scorecard.loc['MAE Matrix'] = 0.
    sign_true = df.sign_true.astype(int)
    sign_pred = df.sign_pred.astype(int)
    
    scorecard.loc['no_of_samples'] = len(target_returns) - evaluate_from
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
    
    for index, row in sign_true.value_counts().iteritems():
        scorecard.loc['sign_true_ratio_' + str(index)] = row / len(sign_true)
    
    for index, row in sign_pred.value_counts().iteritems():
        scorecard.loc['sign_pred_ratio_' + str(index)] = row / len(sign_pred)
    


    # scorecard.loc['ratio_of_classes_y'] = ' / '.join([str(index) + " " + str(round(row / len(sign_true), 2)) for index, row in sign_true.value_counts().iteritems()])
    # scorecard.loc['ratio_of_classes_pred'] = ' / '.join([str(round(row / len(sign_pred), 2)) for index, row in sign_pred.value_counts().iteritems()])


    if method == 'regression':
        scorecard.loc['edge_to_mae'] = scorecard.loc['edge'] / scorecard.loc['MAE']
    elif method == 'classification':
        scorecard.loc['edge_to_mae'] = 0.

    scorecard = scorecard.round(3)
    print("Model name: ", model_name)
    print(scorecard)
    return scorecard  


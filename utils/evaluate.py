from typing import Literal, Callable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from quantstats.stats import skew, sortino
from utils.metrics import probabilistic_sharpe_ratio, sharpe_ratio
from utils.helpers import get_first_valid_return_index
import pandas as pd
import numpy as np

def backtest(returns: pd.Series, signal: pd.Series, transaction_cost = 0.002) -> pd.Series:
    delta_pos = signal.diff(1).abs().fillna(0.)
    costs = transaction_cost * delta_pos
    return (signal * returns) - costs


def __preprocess(target_returns: pd.Series, y_pred: pd.Series, y_true: pd.Series, no_of_classes: Literal['two', 'three-balanced', 'three-imbalanced'], discretize: bool) -> pd.DataFrame:
    y_pred.name = 'y_pred'
    target_returns.name = 'target_returns'
    df = pd.concat([y_pred, target_returns],axis=1).dropna()

    discretize_func = get_discretize_function(no_of_classes)
    # make sure that we evaluate binary/three-way predictions even if the model is a regression
    df['sign_pred'] = df.y_pred.apply(discretize_func) if discretize else df.y_pred
    df['sign_true'] = y_true

    df['result'] = backtest(df.target_returns, df.sign_pred)

    return df

def evaluate_predictions(
                        model_name: str,
                        target_returns: pd.Series,
                        y_pred: pd.Series,
                        y_true: pd.Series,
                        no_of_classes: Literal['two', 'three-balanced', 'three-imbalanced'],
                        print_results: bool,
                        discretize: bool = False,
                        ) -> pd.Series:
    # ignore the predictions until we see a non-zero returns (and definitely skip the first sliding_window_size)
    first_nonzero_return = max(get_first_valid_return_index(target_returns), get_first_valid_return_index(y_pred))
    evaluate_from = first_nonzero_return + 1
    
    target_returns = pd.Series(target_returns[evaluate_from:])
    y_pred = pd.Series(y_pred[evaluate_from:])

    df = __preprocess(target_returns, y_pred, y_true, no_of_classes, discretize)
    
    scorecard = pd.Series()
    
    def count_non_zero(series: pd.Series) -> int:
        return len(series[series != 0])
    no_of_samples = count_non_zero(df.y_pred)
    scorecard.loc['no_of_samples'] = no_of_samples
    sharpe = sharpe_ratio(df.result)
    scorecard.loc['sharpe'] = sharpe
    benchmark_sharpe = sharpe_ratio(df.target_returns)
    scorecard.loc['benchmark_sharpe'] = benchmark_sharpe
    scorecard.loc['prob_sharpe'] = probabilistic_sharpe_ratio(sharpe, benchmark_sharpe, no_of_samples)
    scorecard.loc['sortino'] = sortino(df.result)
    scorecard.loc['skew'] = skew(df.result)

    labels = [1, -1] if no_of_classes == 'two' else [1, -1, 0]
    avg_type = 'weighted' if no_of_classes == 'two' else 'macro'

    if discretize == True:
        scorecard.loc['accuracy'] = accuracy_score(df.sign_true, df.sign_pred) * 100
        scorecard.loc['recall'] = recall_score(df.sign_true, df.sign_pred, labels = labels, average=avg_type)
        scorecard.loc['precision'] = precision_score(df.sign_true, df.sign_pred, labels = labels, average=avg_type)
        scorecard.loc['f1_score'] = f1_score(df.sign_true, df.sign_pred, labels = labels, average=avg_type)
    scorecard.loc['edge'] = df.result.mean()
    scorecard.loc['noise'] = df.y_pred.diff().abs().mean()
    scorecard.loc['edge_to_noise'] = scorecard.loc['edge'] / scorecard.loc['noise']
    
    if discretize == True:
        for index, row in df.sign_true.value_counts().iteritems():
            scorecard.loc['sign_true_ratio_' + str(index)] = row / len(df.sign_true)
        
        for index, row in df.sign_pred.value_counts().iteritems():
            scorecard.loc['sign_pred_ratio_' + str(index)] = row / len(df.sign_pred)

    scorecard = scorecard.round(3)
    if print_results:        
        print("Model name: ", model_name)
        print(scorecard)
    return scorecard  


def __discretize_binary(x): return 1 if x > 0 else -1
def __discretize_threeway(x): return 0 if x == 0 else 1 if x > 0 else -1
def discretize_threeway_threshold(threshold: float) -> Callable:
    def discretize(current_value):
        lower_threshold = -threshold
        upper_threshold = threshold
        if np.isnan(current_value):
            return np.nan 
        elif current_value <= lower_threshold:
            return -1
        elif current_value > lower_threshold and current_value < upper_threshold:
            return 0
        else:
            return 1
    return discretize

def get_discretize_function(no_of_classes: Literal['two', 'three-balanced', 'three-imbalanced']) -> Callable:
    return __discretize_binary if no_of_classes == 'two' else __discretize_threeway



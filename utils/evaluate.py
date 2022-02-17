from typing import Literal, Callable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from quantstats.stats import skew, sortino
from utils.metrics import probabilistic_sharpe_ratio, sharpe_ratio
from utils.helpers import get_first_valid_return_index
import pandas as pd
import numpy as np
from data_loader.types import ForwardReturnSeries, ySeries
from training.types import Stats, WeightsSeries

def backtest(returns: pd.Series, signal: pd.Series, transaction_cost = 0.002) -> pd.Series:
    delta_pos = signal.diff(1).abs().fillna(0.)
    costs = transaction_cost * delta_pos
    return (signal * returns) - costs


def __preprocess(forward_returns: ForwardReturnSeries, y_pred: pd.Series, y_true: pd.Series, no_of_classes: Literal['two', 'three-balanced', 'three-imbalanced'], discretize: bool) -> pd.DataFrame:
    y_pred.name = 'y_pred'
    forward_returns.name = 'forward_returns'
    df = pd.concat([y_pred, forward_returns],axis=1).dropna()

    discretize_func = get_discretize_function(no_of_classes)
    # make sure that we evaluate binary/three-way predictions even if the model is a regression
    df['sign_pred'] = df.y_pred.apply(discretize_func) if discretize else df.y_pred
    df['sign_true'] = y_true

    df['result'] = backtest(df.forward_returns, df.sign_pred)

    return df

def evaluate_predictions(
                        forward_returns: ForwardReturnSeries,
                        y_pred: WeightsSeries,
                        y_true: ySeries,
                        no_of_classes: Literal['two', 'three-balanced', 'three-imbalanced'],
                        discretize: bool = False,
                        ) -> Stats:
    # ignore the predictions until we see a non-zero returns (and definitely skip the first sliding_window_size)
    evaluate_from = max(get_first_valid_return_index(forward_returns), get_first_valid_return_index(y_pred))
    
    forward_returns = pd.Series(forward_returns[evaluate_from:])
    y_pred = pd.Series(y_pred[evaluate_from:])

    df = __preprocess(forward_returns, y_pred, y_true, no_of_classes, discretize)
    
    scorecard = dict()
    
    def count_non_zero(series: pd.Series) -> int:
        return len(series[series != 0])
    no_of_samples = count_non_zero(df.y_pred)
    scorecard['no_of_samples'] = no_of_samples
    sharpe = sharpe_ratio(df.result + 1e-20)
    scorecard['sharpe'] = sharpe
    benchmark_sharpe = sharpe_ratio(df.forward_returns)
    scorecard['benchmark_sharpe'] = benchmark_sharpe
    scorecard['prob_sharpe'] = probabilistic_sharpe_ratio(sharpe, benchmark_sharpe, no_of_samples)
    scorecard['sortino'] = sortino(df.result)
    scorecard['skew'] = skew(df.result)

    labels = [1, -1] if no_of_classes == 'two' else [1, -1, 0]
    avg_type = 'weighted' if no_of_classes == 'two' else 'macro'

    if discretize == True:
        scorecard['accuracy'] = accuracy_score(df.sign_true, df.sign_pred) * 100
        scorecard['recall'] = recall_score(df.sign_true, df.sign_pred, labels = labels, average=avg_type)
        scorecard['precision'] = precision_score(df.sign_true, df.sign_pred, labels = labels, average=avg_type)
        scorecard['f1_score'] = f1_score(df.sign_true, df.sign_pred, labels = labels, average=avg_type)
    scorecard['edge'] = df.result.mean()
    scorecard['noise'] = df.y_pred.diff().abs().mean()
    scorecard['edge_to_noise'] = scorecard['edge'] / (scorecard['noise'] + 0.00001)
    
    if discretize == True:
        for index, row in df.sign_true.value_counts().iteritems():
            scorecard['sign_true_ratio_' + str(index)] = row / len(df.sign_true)
        
        for index, row in df.sign_pred.value_counts().iteritems():
            scorecard['sign_pred_ratio_' + str(index)] = row / len(df.sign_pred)

    scorecard = {k: round(float(v), 3) for k, v in scorecard.items()}
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



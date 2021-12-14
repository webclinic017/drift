from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, r2_score, classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

def format_data_for_backtest(aggregated_data: pd.DataFrame, returns_col: str, only_test_data: pd.DataFrame, preds) -> pd.DataFrame:
    backtest_data = aggregated_data.iloc[-only_test_data.shape[0]:].copy()[returns_col]
    assert backtest_data.shape[0] == only_test_data.shape[0]
    backtest_data = backtest_data.reset_index(drop=True)    
    return pd.concat([backtest_data, pd.Series(preds)], axis='columns')


def evaluate_predictions_regression(model_name: str, y_true, y_pred, sliding_window_size: int):
    evaluate_from = sliding_window_size+1
    y_true = pd.Series(y_true[evaluate_from:-1])
    y_pred = pd.Series(y_pred[evaluate_from:])

    def preprocess(y_true, y_pred):
        y_pred.name = 'y_pred'
        y_true.name = 'y_true'
        df = pd.concat([y_pred, y_true],axis=1).dropna()

        df['sign_pred'] = df.y_pred.apply(np.sign)
        df['sign_true'] = df.y_true.apply(np.sign)
        df['is_correct'] = 0
        df.loc[df.sign_pred * df.sign_true > 0 ,'is_correct'] = 1 # only registers 1 when prediction was made AND it was correct
        df['is_incorrect'] = 0
        df.loc[df.sign_pred * df.sign_true < 0,'is_incorrect'] = 1 # only registers 1 when prediction was made AND it was wrong
        df['is_predicted'] = df.is_correct + df.is_incorrect
        df['result'] = df.sign_pred * df.y_true 
        return df
    
    df = preprocess(y_true, y_pred)
    
    scorecard = pd.Series()
    scorecard.loc['RSQ'] = r2_score(df.y_true,df.y_pred)
    scorecard.loc['MAE'] = mean_absolute_error(df.y_true,df.y_pred)
    scorecard.loc['directional_accuracy'] = df.is_correct.sum()*1. / (df.is_predicted.sum()*1.)*100
    scorecard.loc['edge'] = df.result.mean()
    scorecard.loc['noise'] = df.y_pred.diff().abs().mean()
    scorecard.loc['edge_to_noise'] = scorecard.loc['edge'] / scorecard.loc['noise']
    scorecard.loc['edge_to_mae'] = scorecard.loc['edge'] / scorecard.loc['MAE']
    print("Model name: ", model_name)
    print(scorecard)
    return scorecard  

def evaluate_predictions_classification(model_name: str, y, preds, sliding_window_size: int):
    print("Model: ", model_name)
    evaluate_from = sliding_window_size+1
    y = pd.Series(y[evaluate_from:-1])
    preds = pd.Series(preds[evaluate_from:])
    print(accuracy_score(y, preds))
    print(confusion_matrix(y, preds))
    print(classification_report(y, preds))


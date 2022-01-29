from .types import WeightsSeries, EnsembleOutcome
import pandas as pd
from utils.evaluate import evaluate_predictions
from data_loader.types import ForwardReturnSeries, ySeries
from typing import Literal

def ensemble_weights(
                    input_weights: list[WeightsSeries],
                    forward_returns: ForwardReturnSeries,
                    y: ySeries,
                    no_of_classes: Literal['two', 'three-balanced', 'three-imbalanced'],
                ) -> EnsembleOutcome:
    weights = pd.concat(input_weights, axis=1).mean(axis=1)
    stats = evaluate_predictions(
        forward_returns = forward_returns,
        y_pred = weights,
        y_true = y,
        no_of_classes = no_of_classes,
        print_results = False,
        discretize = True,
    )
    return EnsembleOutcome(weights, stats)

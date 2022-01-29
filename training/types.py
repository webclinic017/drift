import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict

PredictionsSeries = pd.Series
WeightsSeries = pd.Series
ProbabilitiesDataFrame = pd.DataFrame
Stats = Dict[str, float]

ModelOverTime = pd.Series
TransformationsOverTime = list[pd.Series]


@dataclass
class TrainingOutcome:
    model_id: str
    predictions: PredictionsSeries
    probabilities: ProbabilitiesDataFrame
    stats: Stats
    model_over_time: ModelOverTime

@dataclass
class EnsembleOutcome:
    weights: WeightsSeries
    stats: Stats

@dataclass
class BetSizingWithMetaOutcome:
    model_id: str
    meta_training: list[TrainingOutcome]
    meta_transformations: TransformationsOverTime
    weights: WeightsSeries
    stats: Stats

@dataclass
class DirectionalTrainingOutcome:
    training: list[TrainingOutcome]
    transformations: TransformationsOverTime
    
@dataclass
class PipelineOutcome:
    directional_training: DirectionalTrainingOutcome
    bet_sizing: list[BetSizingWithMetaOutcome]
    ensemble: EnsembleOutcome
    secondary_bet_sizing: Optional[BetSizingWithMetaOutcome]

    def get_output_weights(self) -> WeightsSeries:
        return self.secondary_bet_sizing.weights if self.secondary_bet_sizing else self.ensemble.weights

    def get_output_stats(self) -> Stats:
        return self.secondary_bet_sizing.stats if self.secondary_bet_sizing else self.ensemble.stats
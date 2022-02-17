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
    stats: Optional[Stats]
    model_over_time: ModelOverTime


@dataclass
class BetSizingWithMetaOutcome:
    model_id: str
    meta_training: TrainingOutcome
    meta_transformations: TransformationsOverTime
    weights: WeightsSeries
    stats: Optional[Stats]


@dataclass
class DirectionalTrainingOutcome:
    training: TrainingOutcome
    transformations: TransformationsOverTime


@dataclass
class PipelineOutcome:
    directional_training: DirectionalTrainingOutcome
    bet_sizing: BetSizingWithMetaOutcome

    def get_output_weights(self) -> WeightsSeries:
        return self.bet_sizing.weights

    def get_output_stats(self) -> Stats:
        return self.bet_sizing.stats

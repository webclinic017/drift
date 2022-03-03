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
class BaseTrainingOutcome:
    model_id: str
    predictions: PredictionsSeries
    probabilities: ProbabilitiesDataFrame
    model_over_time: ModelOverTime


@dataclass
class TrainingOutcome(BaseTrainingOutcome):
    transformations: TransformationsOverTime
    stats: Optional[Stats]


@dataclass
class BetSizingWithMetaOutcome(TrainingOutcome):
    weights: WeightsSeries


@dataclass
class PipelineOutcome:
    directional_training: TrainingOutcome
    bet_sizing: BetSizingWithMetaOutcome

    def get_output_weights(self) -> WeightsSeries:
        return self.bet_sizing.weights

    def get_output_stats(self) -> Stats:
        return self.bet_sizing.stats

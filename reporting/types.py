from __future__ import annotations
import pandas as pd
from models.base import Model
from typing import Optional, Union


class Reporting:
    def __init__(self):
        self.results: pd.DataFrame = pd.DataFrame() 
        self.all_predictions: pd.DataFrame = pd.DataFrame() 
        self.all_probabilities: pd.DataFrame = pd.DataFrame() 
        self.asset: Reporting.Asset
    
    def get_results(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Reporting.Asset]:
        return self.results, self.all_predictions, self.all_probabilities, self.asset
    

    class Single_Model:
        def __init__(self, model_name: str, model_over_time: pd.Series, transformations_over_time: list[pd.Series]):
            self.model_name: str = model_name
            self.model_over_time: pd.Series = model_over_time
            self.transformations_over_time: list[pd.Series] = transformations_over_time


    class Training_Step: 
        def __init__(self, level: str):
            self.level: str = level
            self.base: list[Reporting.Single_Model] = []
            self.metalabeling: list[list[Reporting.Single_Model]] = []
            
        def get_base(self) -> list[tuple[str, pd.Series, list[pd.Series]]]:
            return [(x.model_name, x.model_over_time, x.transformations_over_time) for x in self.base ]

        def get_metalabeling(self) -> dict:
            structured_dict = dict()
            for i, model in enumerate(self.base):
                structured_dict[model.model_name] = [(x.model_name, x.model_over_time, x.transformations_over_time) for x in self.metalabeling[i]]
            return structured_dict

    class Asset():
        def __init__(self, ticker: str, primary: Reporting.Training_Step, secondary: Reporting.Training_Step):
            self.name: str = ticker
            self.primary: Reporting.Training_Step = primary
            self.secondary: Reporting.Training_Step = secondary 

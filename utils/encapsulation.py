import pandas as pd
from models.base import Model

# | Reporting
# |


class Single_Model:
    def __init__(self, model_name:str, model_over_time:list[Model]):
        self.model_name: str = model_name
        self.model_over_time: list[Model] = model_over_time


class Training_Step: 
    def __init__(self, level:str):
        self.level:str = level
        self.base: list[Single_Model] = []
        self.metalabeling: list[list[Single_Model]] = []


class Asset():
    def __init__(self, ticker:str, primary: Training_Step, secondary: Training_Step):
        self.name:str = ticker
        self.primary:Training_Step = primary
        self.secondary:Training_Step = secondary 


class Reporting:
    def __init__(self):
        self.results:pd.DataFrame = pd.DataFrame() 
        self.all_predictions:pd.DataFrame = pd.DataFrame() 
        self.all_probabilities:pd.DataFrame = pd.DataFrame() 
        self.all_assets:list[Asset] = []
    
    def get_results(self)->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Asset]]:
        return self.results, self.all_predictions, self.all_probabilities, self.all_assets
    


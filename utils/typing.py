from typing import Callable, Union
import pandas as pd


Period = int
IsLogReturn = bool
FeatureExtractor = Callable[[pd.DataFrame, Period, IsLogReturn], Union[pd.DataFrame, pd.Series]]
from typing import Callable, Union, Literal
import pandas as pd

Period = int
IsLogReturn = bool
FeatureExtractor = Callable[[pd.DataFrame, Period], Union[pd.DataFrame, pd.Series]]
Name = str
FeatureExtractorConfig = tuple[Name, FeatureExtractor, list[Period]]
ScalerTypes = Literal['normalize', 'minmax', 'standardize', 'none']

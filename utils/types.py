from typing import Callable, Union, Literal
import pandas as pd

Period = int
IsLogReturn = bool
FeatureExtractor = Callable[[pd.DataFrame, Period, IsLogReturn], Union[pd.DataFrame, pd.Series]]
Name = str
FeatureExtractorConfig = tuple[Name, FeatureExtractor, list[Period]]
Path = str
FileName = str
DataSource = list[tuple[Path, FileName]]
DataCollection = list[DataSource]

ScalerTypes = Literal['normalize', 'minmax', 'standardize', 'none']
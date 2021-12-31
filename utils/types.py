from typing import Callable, Union
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
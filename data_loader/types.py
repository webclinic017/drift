from dataclasses import dataclass
from typing import Literal
import pandas as pd

Path = str
FileName = str


@dataclass
class DataSource:
    path: Path
    file_name: FileName
    freq: Literal["5m", "1h", "1d"]


DataCollection = list[DataSource]

ReturnSeries = pd.Series
ForwardReturnSeries = pd.Series
XDataFrame = pd.DataFrame
ySeries = pd.Series

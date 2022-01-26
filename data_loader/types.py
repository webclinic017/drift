import pandas as pd

Path = str
FileName = str
DataSource = tuple[Path, FileName]
DataCollection = list[DataSource]

ReturnSeries = pd.Series
ForwardReturnSeries = pd.Series
XDataFrame = pd.DataFrame
ySeries = pd.Series
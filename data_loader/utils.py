import pandas as pd

def deduplicate_indexes(df: pd.DataFrame) -> pd.DataFrame: return df[~df.index.duplicated(keep='last')]

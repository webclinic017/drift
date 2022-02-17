import pandas as pd
import numpy as np


def from_df_to_sktime_data(x: pd.DataFrame) -> pd.DataFrame:
    instance_list = []
    instance_list.append([])

    x_data = pd.DataFrame(dtype=np.float32)

    for i in range(len(x.index)):
        instance_list[0].append(pd.Series(x.iloc[i]))

    # only dim_0 for univariate time series
    for dim in range(len(instance_list)):
        x_data["dim_" + str(dim)] = instance_list[dim]

    return x_data

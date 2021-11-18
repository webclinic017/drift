import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

import sys
sys.path.insert(0, '..')

from load_data import load_files

print("success")

def load_format_data(data_dir):
    print("Data Starting    ===>", end="  ")
    # load data
    data = load_files(data_dir, add_features=True, log_returns=False, narrow_format=True)
    # need to treat time as an independent column
    data = data.reset_index().rename({'index':'time'}, axis = 'columns')
    # we need a `time_idx` column for pytorch-forecasting, so we convert the time column to a time index by encoding the dates as consecutive days from the first date.
    data['time_idx'] = (data['time']-data['time'].min()).astype('timedelta64[D]').astype(int)+1
    # volume needs some love before we can use it
    data.drop(columns=['volume'], inplace=True)

    data['month'] = data['month'].astype(str)
    data['day_month'] = data['day_month'].astype(str)
    data['day_week'] = data['day_week'].astype(str)
    
    
    print("<===  Data Loaded")
    print(data.head(3))
    print(data.describe())
    print("")

    
    
    
    return data


def create_dataloaders(data, kwargs):
    print("DataLoader Starting ===>", end="  ")
    # training_cutoff = "YYYY-MM-DD"  # day for cutoff
    # training_cutoff = data["time_idx"].max() - max_prediction_length
    batch_size = kwargs['batch_size']
    del kwargs['batch_size']

    training_dataset = TimeSeriesDataSet(
        data, # data[lambda x: x.date < training_cutoff],
        **kwargs
    )

    #%%
    # create validation and training dataset
    validation = TimeSeriesDataSet.from_dataset(training_dataset, data, predict=True, stop_randomization=True)
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)


    print("<===  DataLoader Created")
    
    
    return training_dataset, train_dataloader, val_dataloader
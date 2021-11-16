#%%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

import sys
sys.path.insert(0, '..')

from load_data import load_files

print("success")

#%%
# load data
data = load_files('../data/', add_features=True, log_returns=False, narrow_format=True)
# need to treat time as an independent column
data = data.reset_index().rename({'index':'time'}, axis = 'columns')
# we need a `time_idx` column for pytorch-forecasting, so we convert the time column to a time index by encoding the dates as consecutive days from the first date.
data['time_idx'] = (data['time']-data['time'].min()).astype('timedelta64[D]').astype(int)+1
# volume needs some love before we can use it
data.drop(columns=['volume'], inplace=True)

data['month'] = data['month'].astype(str)
data['day_month'] = data['day_month'].astype(str)
data['day_week'] = data['day_week'].astype(str)



#%%
# define dataset
max_encoder_length = 36 # this is the look-back window, see https://github.com/jdb78/pytorch-forecasting/issues/448
max_prediction_length = 6
# training_cutoff = "YYYY-MM-DD"  # day for cutoff

# training_cutoff = data["time_idx"].max() - max_prediction_length

#%%
data.head()
data.describe()

#%%
training = TimeSeriesDataSet(
    data, # data[lambda x: x.date < training_cutoff],
    time_idx= 'time_idx',
    target= 'returns',
    # weight="weight",
    group_ids=[ 'ticker' ],
    min_encoder_length = max_encoder_length,##max_encoder_length//2,
    max_encoder_length = max_encoder_length,
    min_prediction_length = 1,
    max_prediction_length = 1,#max_prediction_length,
    static_categoricals=[  ],
    static_reals=[  ],
    time_varying_known_categoricals=[ 'day_month', 'day_week', 'month' ],
    time_varying_known_reals=[ 'vol_10', 'vol_20', 'vol_30', 'vol_60', 'mom_10', 'mom_20', 'mom_30', 'mom_60', 'mom_90'],
    time_varying_unknown_categoricals=[  ],
    time_varying_unknown_reals=[ 'returns' ],
    allow_missing_timesteps=True,
    target_normalizer=GroupNormalizer(
        groups=['ticker'], transformation="softplus")
)

#%%
print(training.index.time)
print(training.index.time.max())

#%%
# create validation and training dataset

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# define trainer with early stopping
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=100,
    gpus=0,
    gradient_clip_val=0.1,
    limit_train_batches=30,
    callbacks=[lr_logger, early_stop_callback],
)

# create the model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=2,
    reduce_on_plateau_patience=4
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
res = trainer.tuner.lr_find(
    tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

# fit the model
trainer.fit(
    tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader,
)


# %%

#%% Import all the stuff, load data, define constants
from sklearn.utils import shuffle
from utils.load_data import load_files, create_target_cum_forward_returns
import pandas as pd
from tensorflow import keras
from utils.normalize import normalize
import tensorflow as tf
from utils.visualize import visualize_loss
from sklearn.preprocessing import MinMaxScaler
from utils.evaluate import print_regression_metrics
import numpy as np
from utils.rolling import rolling_window


data = load_files('data/', add_features=True, log_returns=False)
data.reset_index(drop=True, inplace=True)
data = data[[column for column in data.columns if not column.endswith('volume')]]
data = data[["BTC_returns", "BTC_mom_10", "BTC_mom_20", "BTC_mom_30", "BTC_mom_60", "BTC_vol_10", "BTC_vol_20", "BTC_vol_60", "day_month", "day_week", "month"]]

target_col = 'target'
data = create_target_cum_forward_returns(data, 'BTC_returns', 10)

learning_rate = 0.002
batch_size = 64
epochs = 100

split_fraction = 0.715
train_split = int(split_fraction * int(data.shape[0]))

past = 10
future = 1

start = past + future
end = start + train_split

#%% split data into training - validation sets
train_data = data.loc[0 : train_split - 1]
val_data = data.loc[train_split:]

#%% create features and target for training set & keras dataset
feature_scaler = MinMaxScaler(feature_range= (-1, 1))
target_scaler = MinMaxScaler(feature_range= (-1, 1))

x_train = feature_scaler.fit_transform(train_data.drop(target_col, axis=1).values) # you get the mean and std
y_train = target_scaler.fit_transform(data.iloc[start:end][target_col].values.reshape(-1, 1))

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=past,
    batch_size=batch_size,
)


#%% create features and target for validation set & keras dataset
x_end = len(val_data) - past - future
label_start = train_split + past + future

x_val = feature_scaler.transform(val_data.drop(target_col, axis=1).iloc[:x_end].values) # you use the training data's mean and std
y_val = target_scaler.transform(data.iloc[label_start:][target_col].values.reshape(-1, 1))

dataset_val = keras.utils.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=past,
    batch_size=batch_size,
    shuffle=False,
)


#%%

for batch in dataset_train.take(10):
    batch_inputs, batch_targets = batch

print("Input shape:", batch_inputs.shape)
print("Target shape:", batch_targets.shape)

# print(batch_inputs)
# print(batch_targets)

# %%
model = keras.Sequential()
model.add(keras.layers.LSTM(units = 10, return_sequences = True, activation = 'sigmoid', input_shape=(batch_inputs.shape[1], batch_inputs.shape[2])))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(units = 64, activation = 'sigmoid'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(units = 32, activation = 'sigmoid'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(units = 1))

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss="mean_squared_error")
model.summary()

# %%

path_checkpoint = "model_checkpoint.h5"

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
)

#%%

pred = model.predict(rolling_window(x_val, 11))
pred = pred.reshape(pred.shape[0], 1)
pred = target_scaler.inverse_transform(pred)

print_regression_metrics(y_val, pred)

#%%

visualize_loss(history, "Training and Validation Loss")
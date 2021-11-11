#%% Import all the stuff, load data, define constants
from load_data import load_files
import pandas as pd
from tensorflow import keras
from utils.normalize import normalize
import tensorflow as tf

data = load_files('data/', False)
data.reset_index(drop=True, inplace=True)
data = data[[column for column in data.columns if not column.endswith('volume')]]

ticker_to_predict = 'ETH_returns'

learning_rate = 0.001
batch_size = 128
epochs = 30

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

x_train = normalize(train_data).values
# x_train = train_data.drop(ticker_to_predict, axis=1).values
y_train = normalize(data).iloc[start:end][ticker_to_predict].values

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=past,
    batch_size=batch_size,
)

#%% create features and target for validation set & keras dataset
x_end = len(val_data) - past - future
label_start = train_split + past + future

x_val = normalize(val_data).iloc[:x_end].values
# x_val = val_data.iloc[:x_end].drop(ticker_to_predict, axis=1).values
y_val = normalize(data).iloc[label_start:][ticker_to_predict].values

dataset_val = keras.utils.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=past,
    batch_size=batch_size,
)


#%%

for batch in dataset_train.take(1):
    batch_inputs, batch_targets = batch

print("Input shape:", batch_inputs.shape)
print("Target shape:", batch_targets.shape)


# %%
# model = keras.Sequential()
# model.add(keras.layers.LSTM(units = 50, return_sequences = True, input_shape=(batch_inputs.shape[1], batch_inputs.shape[2])))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(units = 1))

inputs = keras.layers.Input(shape=(batch_inputs.shape[1], batch_inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)
model = keras.Model(inputs=inputs, outputs=outputs)

optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm = 1.)
model.compile(optimizer=optimizer, loss="mean_squared_error")
model.summary()

# %%
path_checkpoint = "model_checkpoint.h5"
# es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

# modelckpt_callback = keras.callbacks.ModelCheckpoint(
#     monitor="val_loss",
#     filepath=path_checkpoint,
#     verbose=1,
#     save_weights_only=True,
#     save_best_only=True,
# )
tf.debugging.enable_check_numerics(
    stack_height_limit=30, path_length_limit=50
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    # callbacks=[modelckpt_callback],
)
# %%

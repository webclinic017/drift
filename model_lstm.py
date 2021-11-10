#%% Import all the stuff, load data, define constants
from load_data import load_files
import pandas as pd
import keras
from utils.normalize import normalize

data = load_files('data/', False)

ticker_to_predict = 'ETH_returns'

learning_rate = 0.001
batch_size = 256
epochs = 10

split_fraction = 0.715
train_split = int(split_fraction * int(data.shape[0]))

past = 720
future = 72

start = past + future
end = start + train_split

#%%

x_train = data.loc[0 : train_split - 1].drop(ticker_to_predict, axis=1).values
y_train = data.iloc[start:end][ticker_to_predict]


dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=past,
    sampling_rate=1,
    batch_size=32,
)

for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)
inputs

# %%
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

# %%
path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)
# %%

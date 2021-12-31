from data_loader.load_data import load_files
import pandas as pd
# from tensorflow import keras
from utils.normalize import normalize
# import tensorflow as tf
from utils.visualize import visualize_loss

from torch.utils.data import DataLoader, random_split
from model_lightning import LitManualAutoEncoder 
import pytorch_lightning as pl

#%%
data = load_files('data/', False)
data.reset_index(drop=True, inplace=True)
data = data[[column for column in data.columns if not column.endswith('volume')]]

data.head()

#%%
ticker_to_predict = 'ETH_returns'

learning_rate = 0.002
batch_size = 64
epochs =  100

split_fraction = 0.715
train_split = int(split_fraction * int(data.shape[0]))

past = 10
future = 1

start = past + future
end = start + train_split


# train = DataLoader(train, batch_size=32)
# test = DataLoader(test, batch_size=32)
# val = DataLoader(val, batch_size=32)


# init model
ae = LitManualAutoEncoder()

# Initialize a trainer
trainer = pl.Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=20)

# Train the model ⚡
# trainer.fit(ae, train, val)
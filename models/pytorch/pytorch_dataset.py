import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].astype(float), self.y[idx].astype(float)

def get_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
    training_data = TimeSeriesDataset(X, y)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader
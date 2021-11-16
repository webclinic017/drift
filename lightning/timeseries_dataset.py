import os
import pandas as pd

import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, file_loader_hook):
        
        df = file_loader_hook()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        pass
        # return image, label
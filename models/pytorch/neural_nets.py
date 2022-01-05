import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import math
import numpy as np

class MultiLayerPerceptron(pl.LightningModule):
    def __init__(self, hidden_layers_ratio: list[float] = [2.0, 2.0], probabilities: bool = False, loss_function=F.mse_loss):
        super().__init__()
        self.hidden_layers_ratio = hidden_layers_ratio
        self.probabilities = probabilities
        self.loss_function = loss_function
        self.float()

    def initialize_network(self, input_dim: int, output_dim: int) -> None:
        self.layers = nn.ModuleList()
        current_dim = input_dim

        for hdim in self.hidden_layers_ratio:
            hidden_layer_size = int(math.floor(current_dim * hdim))
            self.layers.append(nn.Linear(current_dim, hidden_layer_size))
            self.layers.append(nn.ReLU())
            current_dim = hidden_layer_size

        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x: torch.Tensor):
        # in lightning, forward defines the prediction/inference actions
        x = torch.from_numpy(x).float()
        for layer in self.layers:
            x = layer(x)

        if self.probabilities:
            x = F.softmax(x, dim=1)

        return (x.item(), np.array([]))

    def training_step(self, batch: torch.Tensor, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        
        
        loss = 0
        for layer in self.layers:
            x = layer(x.float())

        if self.probabilities:
            p = F.softmax(x, dim=1)
            loss = F.nll_loss(torch.log(p), y.float())

        loss = self.loss_function(x, y.float())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

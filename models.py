'''
Classes for models

TODO:
-1D conv parametrisation
'''
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any
import torch.nn.functional as F

class HackaConv(pl.LightningModule):
    '''
    Parametrisable class for 1d conv following the following architecture from "check paper from Chloe"
    TODO / open questions:
    -How to properly collapse t before fully connected
    '''
    def __init__(self, num_layers=1, num_channels=21, kernel_size=3, lr=1e-4, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([])
        self.losses = []
        self.lr = lr
        for idx in range(num_layers):
            self.layers.append(nn.Conv1d(in_channels=num_channels if idx == 0 else 32 // (2 ** (idx - 1)) * num_channels, out_channels=32 // (2 ** idx) * num_channels, kernel_size=kernel_size, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv1d(in_channels=32 // (2 ** idx) * num_channels, out_channels=32 // (2 ** idx) * num_channels, kernel_size=kernel_size, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool1d(kernel_size=2))

        #TODO: final MaxPool. Check paper
        self.layers.append(nn.Dropout(p=0.1))
        self.layers.append(nn.AvgPool1d(kernel_size=10))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.LazyLinear(out_features=1))
        self.layers.append(nn.Sigmoid()) 

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x * 100 #from 0-1 to 0-100

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = F.mse_loss(z, y)#Binary 
        self.losses.append(loss.detach().cpu().numpy())

        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def set_parameters(self, theta):
        '''
        Manually set parameters using matching theta, not foolproof. To use with functorch
        '''
        p_dict = self.state_dict()
        for p, thet in zip(p_dict, theta):
            p_dict[p] = thet.data
        self.load_state_dict(p_dict)

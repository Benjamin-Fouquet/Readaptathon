import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
#Import Datamodule
from pytorch_lightning import LightningDataModule
from monai.networks.nets import Classifier

class CNNRegressor(nn.Module):
    def __init__(self, in_shape, classes=1, channels=6*[32], strides=6*[2], kernel_size=3, num_res_units=6):
        super(CNNRegressor, self).__init__()
        self.cnn = Classifier(
            in_shape=in_shape,
            classes=classes,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            num_res_units=num_res_units,
            last_act="prelu"
        )
        
        

    def forward(self, x):
        return self.cnn(x)

    

class StupidConvNet(LightningModule):
    def __init__(self, in_shape, classes=1, channels=6*[32], strides=6*[2], kernel_size=3, num_res_units=6, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = CNNRegressor(in_shape, classes, channels, strides, kernel_size, num_res_units)
        self.criterion = nn.L1Loss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer




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
from models import HackaConv


class HackConvPretraining(pl.LightningModule):
    '''
    Parametrisable class for 1d conv following the following architecture from "check paper from Chloe"
    TODO / open questions:
    -How to properly collapse t before fully connected
    '''
    def __init__(self, num_layers=1, num_channels=21, kernel_size=3, lr=1e-4, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layers= HackaConv(num_layers=num_layers, num_channels=num_channels, kernel_size=kernel_size, lr=lr).layers[:-2]
        self.left_layer=nn.LazyLinear(out_features=14)
        self.right_layer=nn.LazyLinear(out_features=14)
        self.lr = lr


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return self.right_layer(x), self.left_layer(x)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        z_l,z_r = self(x)
        loss=[]
        #Create counter of correct predictions
        count_l=0
        count_r=0

        for i in range(y.shape[0]):
            if y[i]>13:
                loss.append(F.cross_entropy(z_l[i:i+1], y[i:i+1]-14))
                if torch.argmax(z_l[i:i+1])==y[i]-14:
                    count_l+=1
                
            elif y[i]<14 and y[i]>0:
                loss.append(F.cross_entropy(z_r[i:i+1], y[i:i+1]))
                if torch.argmax(z_r[i:i+1])==y[i]:
                    count_r+=1

        
        loss=torch.stack(loss).mean()
        self.log("train_loss", loss)
        self.log("train_acc", (count_l+count_r)/y.shape[0])
        print("train_acc", (count_l+count_r)/y.shape[0])
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

        
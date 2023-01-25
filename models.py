'''
Classes for models
'''
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any
import torch.nn.functional as F

class HackaConv(pl.LightningModule):
    '''
    Parametrisable class for 1d conv following the following architecture from "Salami et al.:Using Deep Neural Networks for Human Fall Detection Based on Pose"
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
        return x * 100 #score is converted from a 0-1 scale to a 0-100 scale

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = F.mse_loss(z, y)#Binary 
        self.losses.append(loss.detach().cpu().numpy())

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        z = self(x)
        loss = F.mse_loss(z, y)#Binary 
        self.losses.append(loss.detach().cpu().numpy())

        self.log("val_loss", loss)
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


class HackaConvLSTM(HackaConv):
    '''
    1d conv network with added LSTM. LSTM size is hardcoded to fit our particular datamodule
    '''
    def __init__(self, num_layers=5, num_channels=21, kernel_size=3, lr=0.001, *args: Any, **kwargs: Any) -> None:
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
        self.layers.append(nn.LSTM(input_size=6552, hidden_size=6552, num_layers=2)) #slide needed
        self.layers.append(nn.LazyLinear(out_features=1))
        self.layers.append(nn.Sigmoid()) 


class HackaConvPretraining(pl.LightningModule):
    '''
    Pre-trainable architecture on gesture recognition dataset. First layers can be imported into a HackaConv instance and frozen.
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

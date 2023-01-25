from pretraining_models import HackConvPretraining
from pretraining_graph import Model
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datamodules import BimanualActionsDataModule

model_type='hackconv'

dm=BimanualActionsDataModule(batch_size=32,max_frame=1000)

if model_type=='hackconv':
    model=HackConvPretraining(num_layers=3,num_channels=21,lr=0.001)
elif model_type=='graph':
    model=Model(learning_rate = 1e-2, 
        optimizer = torch.optim.Adam, 
        prefix = 'graph', 
        gpu = 0, 
        in_channels = 1,
        strategy='spatial',criterion=None)
trainer=pl.Trainer(gpus=1)
trainer.fit(model,dm)




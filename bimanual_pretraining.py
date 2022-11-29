from pretraining_models import HackConvPretraining
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datamodules import HackathonDataModule

dm=HackathonDataModule('test/datapath_test','test/scores.json',list(range(1,8)),batch_size=8)
dm.prepare_data()
dm.setup()
dl=DataLoader(dm.pretrain_ds,batch_size=8)
model=HackConvPretraining(num_layers=3,num_channels=21)
trainer=pl.Trainer(gpus=1)
trainer.fit(model,dl)




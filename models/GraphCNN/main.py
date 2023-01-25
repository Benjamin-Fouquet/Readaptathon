#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:07:11 2022

@author: claire
"""

from os.path import expanduser
from posix import listdir
from numpy import NaN, dtype
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import sys
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import glob
import multiprocessing
import math
from pytorch_lightning.loggers import TensorBoardLogger, tensorboard
import argparse

home = expanduser("~")

from G_CNN import Model as G_CNN
from aaha_datamodules import HackathonDataModule


# SET VARIABLES ###############################################################################################
output_path = home+'/Documents/hackathon/Results/'
prefix = 'test_01_G_CNN'
gpu = 0
num_epochs = 100
datapath = '/home/claire/Documents/hackathon/AHA/media/rousseau/Seagate5To/Sync-Data/AHA/derivatives-one-skeleton/'
keypoints = [1, 2, 3, 4, 5, 6, 7]
score_path = "/home/claire/Documents/hackathon/AHA/aha_scores.json"

# NETWORK #####################################################################################################
Net = G_CNN(
    criterion = torch.nn.L1Loss(), 
    learning_rate = 1e-4, 
    optimizer = torch.optim.Adam, 
    prefix = prefix, 
    gpu = 0, 
    in_channels = 1
    )

# DATASET ####################################################################################################
datamodule = HackathonDataModule(datapath, score_path, keypoints, 1)
datamodule.prepare_data()
datamodule.setup()

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()


# TRAINING ###################################################################################################
checkpoint_callback = ModelCheckpoint(filepath=output_path+'Checkpoint_'+prefix+'_{epoch}-{val_loss:.2f}')#, save_top_k=1, monitor=)

logger = TensorBoardLogger(save_dir = output_path, name = 'Test_logger',version=prefix)

trainer = pl.Trainer(
    gpus=[gpu],
    max_epochs=num_epochs,
    progress_bar_refresh_rate=20,
    logger=logger,
    checkpoint_callback= checkpoint_callback,
    precision=16
)

trainer.fit(Net, train_loader, val_loader)
torch.save(Net.state_dict(), output_path+prefix+'_torch.pt')

print('Finished Training')


# # TESTING ###################################################################################################
# I=validation_set[0]
# T=training_set[0]
# score = Net.forward(I)

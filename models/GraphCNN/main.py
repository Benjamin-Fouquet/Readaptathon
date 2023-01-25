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
from datamodules import HackathonDataModule


# SET VARIABLES ###############################################################################################
parser = argparse.ArgumentParser(description='Dynamic MRI Reconstruction')
parser.add_argument('-o', '--output_path', help='InOutput path', type=str, required=False, default = home+'/Documents/hackathon/Results/')
parser.add_argument('-p', '--prefix', help='Prefix', type=str, required=True)
parser.add_argument('-g', '--gpu', help='gpu to use', type=int, required=False, default = 0)
parser.add_argument('-n', '--num_epochs', help='Max number of epochs', type=int, required=False, default=100)
parser.add_argument('-d', '--data_path', help='Data path', type=str, required=False, default='/home/claire/Documents/hackathon/AHA/media/rousseau/Seagate5To/Sync-Data/AHA/derivatives-one-skeleton/')
parser.add_argument('-k', '--keypoints', help='Path to keypoints file', type=list, required=False, default = [1, 2, 3, 4, 5, 6, 7])
parser.add_argument('-s', '--score_path', help='Path to score file', type=str, required=False, default="/home/claire/Documents/hackathon/AHA/aha_scores.json")
parser.add_argument('-S', '--strategy', help='Strategy for the adjacence matrix to use', type=str, required=False, default="spatial")
args = parser.parse_args()

# NETWORK #####################################################################################################
Net = G_CNN(
    criterion = torch.nn.L1Loss(), 
    learning_rate = 1e-4, 
    optimizer = torch.optim.Adam, 
    gpu = 0, 
    in_channels = 1,
    strategy = args.strategy
    )

# DATASET ####################################################################################################
datamodule = HackathonDataModule(args.data_path, args.score_path, args.keypoints, 1)
datamodule.prepare_data()
datamodule.setup()

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()


# TRAINING ###################################################################################################
checkpoint_callback = ModelCheckpoint(filepath=args.output_path+'Checkpoint_'+args.prefix+'_{epoch}-{val_loss:.2f}')#, save_top_k=1, monitor=)

logger = TensorBoardLogger(save_dir = args.output_path, name = 'Test_logger',version=args.prefix)

trainer = pl.Trainer(
    gpus=[args.gpu],
    max_epochs=args.num_epochs,
    progress_bar_refresh_rate=20,
    logger=logger,
    checkpoint_callback= checkpoint_callback,
    precision=16
)

trainer.fit(Net, train_loader, val_loader)
torch.save(Net.state_dict(), args.output_path+args.prefix+'_torch.pt')

print('Finished Training')

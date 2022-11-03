import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datamodules import HackathonDataModule
import torch

dm=HackathonDataModule('test/datapath_test',list(range(1,8)),batch_size=1)
dm.setup()

count=0
#Iterate over all batches
for i,batch in enumerate(dm.train_dataloader()):
    if i==0:
        print(f'Batch {i} : with {len(batch)} variable(s)')

        if len(batch)==1:
            print(f'\t Variable shape : {batch.shape}')
        else:
            for j,v in enumerate(batch):
                print(f'\t Variable {j} : {v.shape}')
    count+=1

print(f'Number of batches : {count}')







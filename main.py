
from os.path import expanduser
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
home = expanduser("~")
import sys
from models import HackaConvNet, HackaConvLSTMNet, HackaConvPretraining, G_CNN
from aaha_datamodules import HackathonDataModule



# SET VARIABLES ###############################################################################################
parser = argparse.ArgumentParser(description='Dynamic MRI Reconstruction')
#DATA
parser.add_argument('-o', '--output_path', help='InOutput path', type=str, required=False, default = home+'/Documents/hackathon/Results/')
parser.add_argument('-d', '--data_path', help='Data path', type=str, required=False, default='data/media/rousseau/Seagate5To/Sync-Data/AHA/derivatives-one-skeleton')
parser.add_argument('-k', '--keypoints', help='Path to keypoints file', type=list, required=False, default = [1, 2, 3, 4, 5, 6, 7])
parser.add_argument('-s', '--score_path', help='Path to score file', type=str, required=False, default="data/aha_scores.json")

#TRAINING
parser.add_argument('-p', '--prefix', help='Experiment name', type=str, required=False)
parser.add_argument('-g', '--gpu', help='gpu to use', type=int, required=False, default = 0)
parser.add_argument('-n', '--num_epochs', help='Max number of epochs', type=int, required=False, default=1)
parser.add_argument('--checkpoint', help='Loading a pretrained model', type=str, required=False, default=None)

#NETWORK
parser.add_argument('--architecture', help='Architecture to use for training (GraphConv, Conv or ConvLSTM)', type=str, required=False, default='Conv')
parser.add_argument('-S', '--strategy', help='Strategy to use for the adjacence matrix', type=str, required=False, default="spatial")
args = parser.parse_args()


# NETWORK #####################################################################################################
if args.architecture == 'GraphConv':
    if args.strategy is None:
        sys.exit("To use GraphConv Net, please specify a strategy for building the adjacency matrix")
    else:
        Net = G_CNN(
            criterion = torch.nn.L1Loss(), 
            learning_rate = 1e-4, 
            optimizer = torch.optim.Adam, 
            gpu = 0, 
            in_channels = 1,
            strategy = args.strategy
            )
elif args.architecture == 'Conv':
    Net = HackaConvNet(
        num_layers=3, 
        num_channels=21, 
        kernel_size=3, 
        lr=1e-4
        )
elif args.architecture == 'ConvLSTM':
    Net = HackaConvLSTMNet(lr=1e-5)
    
else:
    sys.exit("Please specify the network architecture that have to be used for training")
    

# DATASET ####################################################################################################
datamodule = HackathonDataModule(args.data_path, args.score_path, args.keypoints, 1)
datamodule.prepare_data()
datamodule.setup()

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()


# CHECKPOINT LOADING ###################################################################################################
if args.checkpoint == None:
    pass
else:
    if args.checkpoint.split('/')[-1].split('.')[-1]=='pt':
        Net.load_state_dict(torch.load(args.checkpoint))
    elif args.checkpoint.split('/')[-1].split('.')[-1]=='ckpt':
        Net.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    else:
        sys.exit('Entrez un ckeckpoint valide')


# TRAINING ###################################################################################################
checkpoint_callback = None
logger = None
if args.prefix is not None:
    checkpoint_callback = ModelCheckpoint(filepath=args.output_path+'Checkpoint_'+args.prefix+'_{epoch}-{val_loss:.2f}')#, save_top_k=1, monitor=)

    logger = TensorBoardLogger(save_dir = args.output_path, name = 'Test_logger',version=args.prefix)

GPU = [args.gpu] if torch.cuda.is_available() else []

trainer = pl.Trainer(
    gpus=GPU,
    max_epochs=args.num_epochs,
    # progress_bar_refresh_rate=20,
    logger=logger,
    # checkpoint_callback=checkpoint_callback,
    precision=16
)

trainer.fit(Net, train_loader, val_loader)
torch.save(Net.state_dict(), args.output_path+args.prefix+'_torch.pt')

print('Finished Training')

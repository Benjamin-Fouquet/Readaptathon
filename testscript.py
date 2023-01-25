from models import HackaConvNet, HackaConvLSTMNet, HackaConvPretraining #TODO, to change to namespace specific import before git
import aaha_datamodules
import pytorch_lightning as pl
import torch

epochs = 50
gpus = [0] if torch.cuda.is_available() else []
batch_size = 12

datapath = 'data/media/rousseau/Seagate5To/Sync-Data/AHA/derivatives-one-skeleton'
score_path = "data/aha_scores.json"
ckpt = 'data/pretrained_conv1d.ckpt'

dm = aaha_datamodules.HackathonDataModule(datapath=datapath, score_path=score_path, batch_size=batch_size)
dm.prepare_data() #very slow, opti possible?
dm.setup()

train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
# x, y = next(iter(train_loader))

model = HackaConvNet().load_from_checkpoint(checkpoint_path=ckpt, num_layers=3, num_channels=21, kernel_size=3, lr=1e-4, strict=False)

#freeze
# model.freeze()
# for param in model.layers[-2].parameters():
#     param.requires_grad = True

model = HackaConvNet(num_layers=3, num_channels=21, kernel_size=3, lr=1e-4)
trainer = pl.Trainer(gpus=gpus, max_epochs=epochs)
trainer.fit(model, train_loader, val_dataloaders=val_loader)

# model = HackaConvNet(num_layers=5, num_channels=21, kernel_size=3, lr=1e-5)
# trainer = pl.Trainer(gpus=gpus, max_epochs=epochs, log_every_n_steps=10)
# trainer.fit(model, train_loader, val_dataloaders=val_loader)

# model = HackaConvLSTMNet(lr=1e-5)
# trainer = pl.Trainer(gpus=gpus, max_epochs=epochs, log_every_n_steps=10)
# trainer.fit(model, train_loader, val_dataloaders=val_loader)
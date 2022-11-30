from models import (
    HackaConv,
)  # TODO, to change to namespace specific import before git
import datamodules
import pytorch_lightning as pl
import torch

epochs = 50
gpus = [0] if torch.cuda.is_available() else []
batch_size = 12

datapath = "data/AHA/derivatives-one-skeleton"
score_path = "data/aha_scores.json"
ckpt = "lightning_logs/version_22/checkpoints/epoch=999-step=3000.ckpt"

dm = datamodules.HackathonDataModule(
    datapath=datapath, score_path=score_path, batch_size=batch_size
)
dm.prepare_data()  # very slow, opti possible?
dm.setup()

train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
# x, y = next(iter(train_loader))

model = HackaConv().load_from_checkpoint(
    checkpoint_path=ckpt,
    num_layers=3,
    num_channels=21,
    kernel_size=3,
    lr=1e-4,
    strict=False,
)

with torch.no_grad():
    model.eval()
    L2_norm = 0
    L1_norm = 0
    for batch in val_loader:
        x, y = batch
        predict_score = model(x)
        L2_norm += ((predict_score - y) ** 2).sum(dim=0).item()
        L1_norm += (torch.abs(predict_score - y)).sum(dim=0).item()

    print("L2_norm = ", L2_norm)
    print("L1_norm = ", L1_norm)

# freeze
# model.freeze()
# for param in model.layers[-2].parameters():
#     param.requires_grad = True

"""
Datamodules for hackathon project of November 2022
Status:
-Functionnal datamodule for 1 subject

TODO: @Sarah
-Integration of helper functions in dataset 
-Discussion on normalisation
-Interpolation so that all subjects have the same number of points originally 
    -> v1: all subjects match the number of frames from the longest video : done
    -> TODO: v2: interpolate each video w.r.t itself and padd sequence with zero (beginning or end) to match length
-Tests and/or visu on datamodule
"""

import json
import os

from typing import Dict, List, Optional, Tuple, Union
#from isort import file

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from scipy.interpolate import griddata

from interpolation.tool_box import getPoses, interpolate_points_to_video

# number of original keypoints in json files
n_keypoints = 25
# keypoints to keep
keypoints = [1, 2, 3, 4, 5, 6, 7]
# datapath    = "/home/benjamin/Documents/hackathon/AHA/derivatives-one-skeleton/"
datapath = "/home/reynaudsarah/Documents/Data/hackathon/AHA/derivatives-one-skeleton"
file_path = f"{datapath}/020101_aha_j0.json"




def get_last_timestp(folder, verbose=True):
    files = os.listdir(folder)
    if len(files) > 0:
        files.sort()
        last_timestp = int(files[-1].split("_")[1])
        return last_timestp
    else:
        if verbose:
            print(f"{folder} is empty")
    return folder



class HackathonDataset(Dataset):
    def __init__(self, datapath: str , keypoints) -> None:
        """
        Args:
            datapath (str): path to the data
            keypoints (list): list of keypoints to keep
        """
        super().__init__()
        self.datapath: str = datapath
        self.keypoints = keypoints
        self.scores = 0


        last_timestp = []
        empty_folders = []

        for folder in os.listdir(datapath):
            l_timestp = get_last_timestp(f"{datapath}/{folder}", verbose=False)
            if type(l_timestp) is int:
                last_timestp.append(get_last_timestp(f"{datapath}/{folder}"))
            else:
                empty_folders.append(l_timestp)

        max_timestp = np.max(np.array(last_timestp))
        starter_tensor = np.empty((1, max_timestp, len(keypoints), 3), dtype=np.float32)
        subjects = []

        for folder in list(os.listdir(datapath)):
            if f"{datapath}/{folder}" not in empty_folders:
                subjects.append(folder)
                frame_points = getPoses(f"{datapath}/{folder}")
                interp_frame_points = interpolate_points_to_video(frame_points, None)
                #Pad with zeros to match max_timestp
                tensor2 = np.zeros((1, max_timestp, len(keypoints), 3), dtype=np.float32)
                tensor2[0,:interp_frame_points.shape[0], :, :] = interp_frame_points[:,self.keypoints,1:]
                starter_tensor = np.concatenate((starter_tensor, tensor2), axis=0)

        subjects_tensor = starter_tensor[1:, ...]
        subjects_tensor = (
            subjects_tensor.reshape(
                (subjects_tensor.shape[0], subjects_tensor.shape[1], len(keypoints) * 3)
            )
        ).swapaxes(1, 2)

        self.tensor = torch.FloatTensor( subjects_tensor )
        self.subjects = subjects

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index) -> None:
        # 1 subject, 1 score
        return self.tensor[index], self.scores[index] if self.scores else None


class HackathonDataModule(pl.LightningDataModule):
    def __init__(
        self, datapath: str=None, keypoints = list( np.arange(1,8, dtype=int) ), batch_size: int = 1
    ):
        super().__init__()
        self.datapath = datapath
        self.keypoints = keypoints
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        self.dataset = HackathonDataset(self.datapath, self.keypoints)
        return super().prepare_data()

    def setup(self, split: float = 0.2) -> None:
        self.prepare_data()
        self.dataset.tensor = (self.dataset.tensor - torch.min(self.dataset.tensor)) / (
            torch.max(self.dataset.tensor) - torch.min(self.dataset.tensor)
        )
        self.val_ds, self.train_ds = torch.split(
            self.dataset.tensor,
            [
                int(split * len(self.dataset.tensor)),
                len(self.dataset.tensor) - int(split * len(self.dataset.tensor)),
            ],
        )
        self.test_ds = None
        # reflechir sur norm, best approach prob. {xi}, {yi} min/max. Pas touche les C TODO

        return super().setup()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, self.batch_size, shuffle=True, num_workers=int(os.cpu_count()/2)
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, self.batch_size, shuffle=False, num_workers=int(os.cpu_count()/2)
        )

    def test_dataloader(self) -> DataLoader:
        return None







'''
-> Plus besoin avec les fonctions de Nathan
def sortFn(e):
    # function to sort a list of file names by frame id
    return e["frame_id"]


def get_frames(folder: str):
    """
    Get all the file names of files in folder,
    sort by frame id (utile?) and return the sorted list of files and
    the number of frames
    """
    # get file names in the folder
    file_list = []
    file_names = os.listdir(folder)

    # identify the frame id and create a list of file name and frame id/number
    # to be able to sort the list by frame id/number
    for f in file_names:
        frame_id = f.split("_")[1]
        file_list.append({"file_name": f"{folder}/{f}", "frame_id": int(frame_id)})

    # sort the file names by frame number
    file_list.sort(key=sortFn)

    n_frames = len(file_list)

    return file_list, n_frames


def concatenate_frames(frame_list: list, n_keypoints: int):
    """
    concatenate the pose data of the frame files in frame_list for each points.
    Returns a dictionnary with the sequence of 2d values for each point of the skeleton (pose_dict)
    and a dictionnary with the sequence of confidence values for each point of the skeleton (confidence_dict)

    model of keys: p0, p1, ..., p24 (points 0 to 24)
    """

    n_frames = len(frame_list)
    pose = []

    for k in range(0, n_frames):

        # open the file of the frame
        file = frame_list[k]["file_name"]
        with open(file, "r") as f:
            dic = json.load(f)["people"][0]
        # add the data of the frame to the data of previous frames
        if len(pose) == 0:
            pose = dic["pose_keypoints_2d"]
        else:
            pose += dic["pose_keypoints_2d"]
    n_points = len(pose)

    # concatenate data properly (by points)
    # pose_dict = dictionnary with the points data sequence per keypoints
    # confidence_dict = dictionnary of the confidence value sequence per keypoints
    pose_dict = {f"p{k}": [] for k in range(0, n_keypoints)}
    confidence_dict = {f"conf_p{k}": [] for k in range(0, n_keypoints)}

    # for each keypoint
    tensor = np.zeros((3 * n_keypoints, n_frames), dtype=np.float32)
    for k in range(n_keypoints):
        tensor[k, :] = np.array([pose[i] for i in range(k, n_points, 3 * n_keypoints)])
        tensor[k + 1, :] = np.array(
            [pose[i + 1] for i in range(k, n_points, 3 * n_keypoints)]
        )
        tensor[k + 2, :] = np.array(
            [pose[i + 2] for i in range(k, n_points, 3 * n_keypoints)]
        )
    return tensor


class HackathonDataset(Dataset):
    def __init__(self, tensor, subjects: None) -> None:
        super().__init__()
        self.tensor = torch.FloatTensor(tensor)
        self.subjects: str = subjects
        self.scores = 0

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index) -> None:
        # 1 subject, 1 score
        return self.tensor[index, ...], self.scores[index] if self.scores else None


class HackathonDataModule(pl.LightningDataModule):
    def __init__(
        self, tensor: torch.FloatTensor = None, subjects=None, batch_size: int = 1
    ):
        super().__init__()
        self.tensor = torch.FloatTensor(tensor)
        self.subjects = subjects
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        self.dataset = HackathonDataset(self.tensor, self.subjects)
        return super().prepare_data()

    def setup(self, split: float = 0.2) -> None:
        self.tensor = (self.tensor - torch.min(self.tensor)) / (
            torch.max(self.tensor) - torch.min(self.tensor)
        )
        self.val_ds, self.train_ds = torch.split(
            self.tensor,
            [
                int(split * len(self.tensor)),
                len(self.tensor) - int(split * len(self.tensor)),
            ],
        )
        self.test_ds = None
        # reflechir sur norm, best approach prob. {xi}, {yi} min/max. Pas touche les C TODO

        return super().setup()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, self.batch_size, shuffle=True, num_workers=os.cpu_count()
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, self.batch_size, shuffle=False, num_workers=os.cpu_count()
        )

    def test_dataloader(self) -> DataLoader:
        return None
'''


'''
=> put in the dataset part.
last_timestp = []
empty_folders = []

for folder in os.listdir(datapath):
    l_timestp = get_last_timestp(f"{datapath}/{folder}")
    if type(l_timestp) is int:
        last_timestp.append(get_last_timestp(f"{datapath}/{folder}"))
    else:
        empty_folders.append(l_timestp)

max_timestp = np.max(np.array(last_timestp))
starter_tensor = np.zeros((1, max_timestp, len(keypoints), 3), dtype=np.float32)
subjects = []

for folder in os.listdir(datapath):
    if f"{datapath}/{folder}" not in empty_folders:
        subjects.append(folder)
        frame_points = getPoses(f"{datapath}/{folder}")
        interp_frame_points = interpolate_points_to_video(frame_points, max_timestp)
        tensor2 = np.expand_dims(interp_frame_points[:, keypoints, 1:], 0)
        starter_tensor = np.concatenate((starter_tensor, tensor2), axis=0)

subjects_tensor = starter_tensor[1:, ...]
subjects_tensor = (
    subjects_tensor.reshape(
        (subjects_tensor.shape[0], subjects_tensor.shape[1], len(keypoints) * 3)
    )
).swapaxes(1, 2)

datamodule = HackathonDataModule(subjects_tensor, subjects, 1)
datamodule.prepare_data()
datamodule.setup()
'''

###############################

if __name__ == "__main__": 
    datamodule = HackathonDataModule(datapath, keypoints, 1)
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

## TODO: vizu




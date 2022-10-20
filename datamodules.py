"""
Datamodules for hackathon project of November 2022
Status:
-Functionnal datamodule for 1 subject

TODO: @Sarah
-Integration of helper functions in dataset 
-Discussion on normalisation
-Interpolation so that all subjects have the same number of points originally
-Tests and/or visu on datamodule
"""

import json
import os

from typing import Dict, Optional, Tuple, Union, list

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


# number of original keypoints in json files
n_keypoints = 25
datapath = "/home/benjamin/Documents/hackathon/AHA/derivatives-one-skeleton/"


def find_char(s, ch):
    """
    return all occurence of a given char ch in a string s
    inputs :
       s : string to search in
      ch : char to find in s
    output :
      list of index of ch char
    """
    return [i for i, ltr in enumerate(s) if ltr == ch]


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
        tirets = find_char(f, "_")
        frame_id = f[tirets[0] + 1 : tirets[1]]

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
        # Not sure that the part dim==3 is useful as we don't have any info on the third space dimension
        """
        if dim == 3:
            pose_dict[f"p{k}"] = [
                (pose[k], pose[k+1], 0) for k in range(k,n_points,3*n_keypoints)
            ]

        else:
        """

        tensor[k, :] = np.array([pose[i] for i in range(k, n_points, 3 * n_keypoints)])
        tensor[k + 1, :] = np.array(
            [pose[i + 1] for i in range(k, n_points, 3 * n_keypoints)]
        )
        tensor[k + 2, :] = np.array(
            [pose[i + 2] for i in range(k, n_points, 3 * n_keypoints)]
        )
        """
        pose_dict[f"p{k}"] = [ 
            (pose[i], pose[i+1]) for i in range(k,n_points,3*n_keypoints)
        ]
        confidence_dict[f"conf_p{k}"] = [
            pose[i+2] for i in range(k,n_points,3*n_keypoints)
        ]"""

    return tensor  # pose_dict, confidence_dict


starter_tensor = np.zeros((1, 18, 38712), dtype=np.float32)

for filepath in os.listdir(datapath):
    nfiles, n_frames = get_frames(datapath + filepath)

    tensor = concatenate_frames(frame_list=nfiles, n_keypoints=n_keypoints)
    tensor2 = np.expand_dims(tensor[3 : 3 * 7, :], 0)
    starter_tensor = np.concatenate((starter_tensor, tensor2), axis=0)

subjects_tensor = starter_tensor[1:, ...]


class HackathonDataset(Dataset):
    def __init__(self, tensor) -> None:
        super().__init__()
        self.tensor = torch.FloatTensor(tensor)
        self.subjects: str = None
        self.scores: list[int, ...] = 0

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index) -> None:
        # 1 subject, 1 score
        return self.tensor[index, ...], self.scores[index] if self.scores else None


class HackathonDataModule(pl.LightningDataModule):
    def __init__(self, tensor: torch.FloatTensor = None, batch_size: int = 1):
        super().__init__()
        self.tensor = torch.FloatTensor(tensor)
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        self.dataset = HackathonDataset(self.tensor)
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

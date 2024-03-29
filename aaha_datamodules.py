"""
Datamodules for hackathon project of November 2022
- AHA evaluation dataset from Ildys
- Bimanual action dataset from: https://bimanual-actions.humanoids.kit.edu/
"""

import json
import os
import threading
import time
from functools import wraps
from sklearn import preprocessing

from typing import Dict, List, Optional, Tuple, Union

# from isort import file

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from scipy.interpolate import griddata

from interpolation.tool_box import (
    getPoses,
    interpolate_points_to_video,
)
from torch.utils.data import ConcatDataset
from interpolation.interpolation import interpolate, remove_anomalies

n_keypoints = 25 # number of original keypoints in json files
keypoints = [1, 2, 3, 4, 5, 6, 7] # keypoints to keep
datapath = (
    "/home/reynaudsarah/Documents/Data/hackathon/AHA/derivatives-one-skeleton"
)
score_path = "/home/reynaudsarah/Documents/Data/hackathon/AHA/aha_scores.json"
file_path = f"{datapath}/020101_aha_j0.json"



def get_last_timestp(folder, verbose=True):
    """Get the last timestep of a video (if poses were extracted) 
    
    Args: 
        folder: the folder in which the different files of a video are stored.
        verbose: verbose attribute
    Returns: 
        last_timestep if poses were extracted (folder not empty)
        folder if no poses were extracted (folder empty)
    """
    files = os.listdir(folder)
    if len(files) > 0:
        files.sort()
        last_timestp = int(files[-1].split("_")[1])
        return last_timestp
    else:
        if verbose:
            print(f"{folder} is empty")
        return folder


def interp_clean(y, window_length=31, poly_order=3, threshold=150):
    """Interpolate data points and remove anomalies above a given threshold.

    Args:
        y: Data points.
        window_length: Window length used during interpolation. Defaults to 31.
        poly_order: Polynomial order used during interpolation. Defaults to 3.
        threshold: Threshold above which anomalies are removed. Defaults to 150.

    Returns:
        A tuple of interpolated data points and data points with anomalies removed.
    """
    y_interp = interpolate(
        y, window_length=window_length, poly_order=poly_order
    )
    y_clean = remove_anomalies(y, y_interp, threshold)

    return y_interp, y_clean


def smooth_tensor(full_tensor):
    """Smooth data points by removing anomalies.

    Args:
        full_tensor: Data points from which anomalies are removed.
    """
    for i, patient in enumerate(full_tensor):
        for j, keypoint in enumerate(patient):
            k_interp, k_clean = interp_clean(keypoint)
            full_tensor[i, j, :] = torch.tensor(k_interp)


def normalize_tensor(full_tensor):
    """Normalize a tensor. The mean of every patients are set to 0.
       The standard deviation is set to 1 over all patients.

    Args:
        full_tensor: a tensor of shape (nb_patients, 3 * keypoints, nb_frames)
    """
    xi = full_tensor[:, ::3, ...]
    yi = full_tensor[:, 1::3, ...]
    x_shape, y_shape = xi.shape, yi.shape

    mean_x = (xi.reshape(xi.shape[0], -1)).mean(axis=1)
    mean_y = (yi.reshape(yi.shape[0], -1)).mean(axis=1)

    xi -= mean_x.unsqueeze(-1).unsqueeze(-1)
    yi -= mean_y.unsqueeze(-1).unsqueeze(-1)

    xi = xi.reshape(x_shape)
    yi = yi.reshape(y_shape)

    z = torch.stack((xi, yi), dim=0)
    z_shape = z.shape
    z = z.reshape(2, -1).transpose(-1, 0)
    scaler = preprocessing.StandardScaler().fit(z)
    z_scaled = scaler.transform(z)
    z_scaled = z_scaled.transpose(-1, 0).reshape(z_shape)
    full_tensor[:, ::3, ...] = torch.tensor(
        z_scaled[0, ...], dtype=full_tensor.dtype
    )
    full_tensor[:, 1::3, ...] = torch.tensor(
        z_scaled[1, ...], dtype=full_tensor.dtype
    )


class HackathonDataset(Dataset):
    """ 
    Creates a dataset of (subject, AHA score) from 
    - datapath: path to the folder containing all the subject folders.
    - score_path: path to the score file.
    - keypoints: keypoints of pose to keep for analysis.
    """
    def __init__(self, datapath: str, score_path: str, keypoints) -> None:
        super().__init__()
        self.datapath: str = datapath
        self.score_path: str = score_path
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
        self.max_timestp = max_timestp
        starter_tensor = np.zeros(
            (1, max_timestp, len(keypoints), 3), dtype=np.float32
        )
        subjects = []

        for folder in os.listdir(datapath):
            if f"{datapath}/{folder}" not in empty_folders:
                subjects.append(folder)

                frame_points = getPoses(f"{datapath}/{folder}")
                interp_frame_points = interpolate_points_to_video(frame_points)

                padded_interp_frame_points = np.zeros(
                    (1, max_timestp, len(keypoints), 3), dtype=np.float32
                )
                padded_interp_frame_points[
                    0, : interp_frame_points.shape[0], ...
                ] = interp_frame_points[:, keypoints, 1:] # padding to have equal length sequences.

                starter_tensor = np.concatenate(
                    (starter_tensor, padded_interp_frame_points), axis=0
                ) 

        subjects_tensor = starter_tensor[1:, ...]
        subjects_tensor = (
            subjects_tensor.reshape(
                (
                    subjects_tensor.shape[0],
                    subjects_tensor.shape[1],
                    len(keypoints) * 3,
                )
            )
        ).swapaxes(1, 2)

        self.tensor = torch.FloatTensor(subjects_tensor)

        self.subjects = subjects

        # loading scores
        self.scores = torch.zeros((len(subjects), 1), dtype=torch.float32)
        with open(self.score_path, "r") as f:
            score_dict = json.load(f)

        i = 0
        for s in subjects:
            self.scores[i] = float(score_dict[s])
            i += 1
        self.scores = self.scores.squeeze(0)

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, index) -> None:
        # 1 subject, 1 score
        return (
            self.tensor[index, ...],
            self.scores[index],
        )  

    def clean(self):
        '''
        Remove subjects without score for training
        '''
        mask = (self.scores > -1) * 1.0
        self.scores = self.scores * mask
        self.tensor = self.tensor[self.scores.nonzero()[:,0]]
        self.scores = self.scores[self.scores.nonzero()[:,0]]

class HackathonDataModule(pl.LightningDataModule):
    """ 
    pytorch lightning datamodule.
    datapath, score_path and keypoints -> for the HackathonDataset
    """
    def __init__(
        self,
        datapath: str = None,
        score_path: str = None,
        keypoints=list(np.arange(1, 8, dtype=int)),
        batch_size: int = 1,
        shuffle_dataset: bool = True,
        normalize=False,
        smooth=False,
    ):
        super().__init__()
        self.datapath = datapath
        self.score_path = score_path
        self.keypoints = keypoints
        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset
        self.normalize = normalize
        self.smooth = smooth

    def prepare_data(self) -> None:
        self.dataset = HackathonDataset(
            self.datapath, self.score_path, self.keypoints
        )
        self.dataset.clean()
        if self.normalize:
            normalize_tensor(self.dataset.tensor)
        if self.smooth:
            smooth_tensor(self.dataset.tensor)
        return super().prepare_data()

    def setup(self, split: float = 0.2) -> None:
        # create indices to split dataset
        indices = list(range(len(self.dataset)))
        if self.shuffle_dataset:
            np.random.shuffle(indices)
        train_indices, val_indices = (
            indices[int(split * len(self.dataset)) :],
            indices[: int(split * len(self.dataset))],
        )
        if len(indices) == 1:
            print("len indices = 1")
            self.train_ds, self.val_ds = self.dataset, self.dataset
        else:
            self.train_ds = [self.dataset[k] for k in train_indices]
            self.val_ds = [self.dataset[k] for k in val_indices]
        """
        self.pretrain_ds = get_bimanual_actions_dataset(
            max_frame=self.dataset.max_timestp
        )"""
        """
        # avec des sampler? -> si oui ajouter argument dans dataloaders
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        self.val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
        """

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

    def test_dataloader(self) -> DataLoader:
        return None


class HackathonDatasetFromTensor(Dataset):
    """ 
    Dataset of (subject pose,score) of the AHA evaluation. From pre-loaded and save tensors.
    To preload tensor use convert_data_to_tensor(...) function.
    """
    def __init__(self, datapath: str, keypoints) -> None:
        """ 
        Args: 
            datapath: path of the folder in which pose tensor were saved
            keypoints: keypoints of pose to use
        """
        super().__init__()
        self.datapath: str = datapath
        self.keypoints = keypoints
        self.subjects = []
        
        scores = []
        data = []

        for folder in os.listdir(datapath):
            
            d = torch.load(f"{datapath}/{folder}")

            if d["score"]>0: 
                self.subjects.append(folder)
                data.append(d["tensor"][keypoints[0]*3 : (keypoints[-1]+1)*3])
                scores.append( float( d["score"] ) )


        self.tensor = torch.cat( [dt.unsqueeze(0) for dt in data] )
        self.scores = torch.tensor( scores )

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index) -> None:
        # 1 subject, 1 score
        return (
            self.tensor[index, ...],
            self.scores[index],
        ) 

class HackathonDataModuleFromTensor(pl.LightningDataModule):
    """ 
    Pytorch lightning datamodule for the dataset HackthonDatasetFromTensor
    """
    def __init__(
        self,
        datapath: str = None,
        keypoints=list(np.arange(1, 8, dtype=int)),
        batch_size: int = 1,
        shuffle_dataset: bool = True,
    ):
        super().__init__()
        self.datapath = datapath
        self.keypoints = keypoints
        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset

    def prepare_data(self) -> None:
        self.dataset = HackathonDatasetFromTensor(
            self.datapath, self.keypoints
        )
        return None

    def setup(self, split: float = 0.2) -> None:
        # create indices to split dataset
        indices = list(range(len(self.dataset)))
        if self.shuffle_dataset:
            np.random.shuffle(indices)
        train_indices, val_indices = (
            indices[int(split * len(self.dataset)) :],
            indices[: int(split * len(self.dataset))],
        )
        if len(indices) == 1:
            print("len indices = 1")
            self.train_ds, self.val_ds = self.dataset, self.dataset
        else:
            self.train_ds = [self.dataset[k] for k in train_indices]
            self.val_ds = [self.dataset[k] for k in val_indices]

        self.test_ds = None
        # reflechir sur norm, best approach prob. {xi}, {yi} min/max. Pas touche les C TODO

        return None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

    def test_dataloader(self) -> DataLoader:
        return None


def convert_data_to_tensor( datapath: str, score_path:str, store_tensor_folder:str):
    """ 
    Converts json files of pose extraction of videos in datapath to tensor.
    Saves tensor and scores for each subject (dictionary). 

    Args: 
        datapath: path of the fodler containing the pose extractions
        score_path: path to the score file
        store_tensor_folder: folder in which to save the tensor and score ".pt" files

    Returns:
        subjects: list of the folder containing data of pose extraction
   
    """

    last_timestp    = []
    empty_folders   = []
    subjects        = []

    for folder in os.listdir(datapath):
        l_timestp = get_last_timestp(f"{datapath}/{folder}", verbose=False)
        if type(l_timestp) is int:
            subjects.append(folder)
            last_timestp.append(get_last_timestp(f"{datapath}/{folder}"))
        else:
            empty_folders.append(l_timestp)
    
    max_timestp = np.max(np.array(last_timestp))
    max_timestp = max_timestp
    
    # loading scores
    with open(score_path, "r") as f:
        score_dict = json.load(f)

    for folder in os.listdir(datapath):

        if f"{datapath}/{folder}" not in empty_folders:
            subject = folder

            frame_points = getPoses(f"{datapath}/{folder}")
            
            interp_frame_points = interpolate_points_to_video(frame_points)

            padded_interp_frame_points = np.zeros(
                (1, max_timestp, n_keypoints, 3), dtype=np.float32
            )
            padded_interp_frame_points[
                0, : interp_frame_points.shape[0], ...
            ] = interp_frame_points[:, :, 1:]

            

            tmp_tensor = torch.from_numpy(
                padded_interp_frame_points.reshape(
                    (
                        max_timestp,
                        n_keypoints * 3,
                    )
                )
            ).swapaxes(0, 1)

            score = score_dict[subject]


            tensor_dict = {"tensor":tmp_tensor, "score":score}
            torch.save(tensor_dict, f"{store_tensor_folder}/{subject}.pt")
    
    return subjects


###############################

if __name__ == "__main__":
    # datamodule = HackathonDataModule(datapath, score_path, keypoints, 1)
    # datamodule.prepare_data()
    # datamodule.setup()

    # train_loader = datamodule.train_dataloader()
    # val_loader = datamodule.val_dataloader()

    open_pose_files = f"{datapath}/derivatives-one-skeleton"
    store_tensor_folder = f"{datapath}/tensors"

    # This must be done before using the version of the datamodule that loads the tensors.
    # Only pre-processing done on the recorded tensors: frame interpolation
    if not os.path.exists(store_tensor_folder): 
        os.mkdir(store_tensor_folder)
        _ , ifp= convert_data_to_tensor(datapath=open_pose_files, score_path=score_path, store_tensor_folder=store_tensor_folder)

    datamodule = HackathonDataModuleFromTensor(store_tensor_folder, keypoints, 1)
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    ## TODO : some view data.

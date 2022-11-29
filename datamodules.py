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
    getPosesBM,
)
from torch.utils.data import ConcatDataset
from interpolation.interpolation import interpolate, remove_anomalies


def async_loader(func):
    """Run a function in parallel"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


# number of original keypoints in json files
n_keypoints = 25
# keypoints to keep
keypoints = [1, 2, 3, 4, 5, 6, 7]
# datapath    = "/home/benjamin/Documents/hackathon/AHA/derivatives-one-skeleton/"
datapath = (
    "/home/reynaudsarah/Documents/Data/hackathon/AHA/derivatives-one-skeleton"
)
score_path = "/home/reynaudsarah/Documents/Data/hackathon/AHA/aha_scores.json"

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


def interp_clean(y, window_length=31, poly_order=3, threshold=150):
    y_interp = interpolate(
        y, window_length=window_length, poly_order=poly_order
    )
    y_clean = remove_anomalies(y, y_interp, threshold)

    return y_interp, y_clean


def smooth_tensor(full_tensor):
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
                ] = interp_frame_points[:, keypoints, 1:]

                starter_tensor = np.concatenate(
                    (starter_tensor, padded_interp_frame_points), axis=0
                )  # tensor2

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
        return len(self.subjects)

    def __getitem__(self, index) -> None:
        # 1 subject, 1 score
        return (
            self.tensor[index, ...],
            self.scores[index],
        )  # if self.scores else None

    def clean(self):
        '''
        Remove subjects without score for training
        '''
        mask = (self.scores > -1) * 1.0
        self.scores = self.scores * mask
        self.tensor = self.subjects[self.scores.nonzero()]
        self.scores = self.subjects[self.scores.nonzero()]

class BimanualActionsDataset(Dataset):
    def __init__(self, take_folder: str, gt_file, max_frame: int) -> None:
        super().__init__()
        self.take_folder: str = take_folder
        self.gt_file = gt_file
        self.max_frame = max_frame
        self.scores = 0
        self.keypoints = [
            "Neck",
            "RShoulder",
            "RElbow",
            "RWrist",
            "LShoulder",
            "LElbow",
            "LWrist",
        ]
        self.points = getPosesBM(
            folder=self.take_folder, keypoints=self.keypoints
        )
        with open(self.gt_file, "r") as f:
            self.gt_dict = json.load(f)

        right_hand_gt = self.gt_dict["right_hand"]
        left_hand_gt = self.gt_dict["left_hand"]
        right_hand_tasks = right_hand_gt[1::2]
        left_hand_tasks = left_hand_gt[1::2]
        right_hand_tmstps = right_hand_gt[::2]
        left_hand_tmstps = left_hand_gt[::2]
        actions_points = []
        frame_to_remove = {"right_hand": [], "left_hand": []}
        for i in range(len(right_hand_tasks)):
            if right_hand_tasks[i] == None:
                frame_to_remove["right_hand"].append(right_hand_tmstps[i])
            else:
                padded_action_point = np.empty(
                    (self.points.shape[0], max_frame)
                )
                action_point = self.points[
                    :, right_hand_tmstps[i] : right_hand_tmstps[i + 1]
                ]
                padded_action_point[:, : action_point.shape[1]] = action_point
            actions_points.append(padded_action_point)
        for i in range(len(left_hand_tasks)):
            if left_hand_tasks[i] == None:
                frame_to_remove["left_hand"].append(left_hand_tmstps[i])
            else:
                padded_action_point = np.empty(
                    (self.points.shape[0], max_frame)
                )
                action_point = self.points[
                    :, left_hand_tmstps[i] : left_hand_tmstps[i + 1]
                ]
                padded_action_point[:, : action_point.shape[1]] = action_point
                actions_points.append(padded_action_point)
        self.actions_points = torch.FloatTensor(np.stack(actions_points))
        # Remove frames with no action

        right_hand_tasks = [x for x in right_hand_tasks if x != None]
        left_hand_tasks = [x for x in left_hand_tasks if x != None]

        self.actions_gt = torch.FloatTensor(
            np.concatenate((right_hand_tasks, left_hand_tasks))
        )

    def __len__(self):
        return len(self.actions_points)

    def __getitem__(self, index):
        return self.actions_points[index : index + 1], self.actions_gt[index]

        # print(self.points.shape)


threads = []


def get_bmdataset(take_folder, gt_file, max_frame):
    threads.append(0)
    ds = BimanualActionsDataset(take_folder, gt_file, max_frame)
    threads.pop()
    return ds


def get_bimanual_actions_dataset(
    max_frame, root_dir="F:\\bimacs_derived_data_body_pose\\"
):
    """Return a dataset of bimanual actions"""
    data_dir = os.path.join(root_dir, "bimacs_derived_data")
    gt_dir = os.path.join(root_dir, "bimacs_rgbd_data_ground_truth")
    takes = []
    flag = False
    for sub_folder in os.listdir(data_dir):
        for task_folder in os.listdir(os.path.join(data_dir, sub_folder)):
            for take_folder in os.listdir(
                os.path.join(data_dir, sub_folder, task_folder)
            ):
                gt_file = os.path.join(
                    gt_dir, sub_folder, task_folder, take_folder + ".json"
                )
                if not flag:
                    takes.append(
                        get_bmdataset(
                            os.path.join(
                                data_dir,
                                sub_folder,
                                task_folder,
                                take_folder,
                                "body_pose",
                            ),
                            gt_file,
                            max_frame,
                        )
                    )
                    flag = True

    return ConcatDataset(takes)


class HackathonDataModule(pl.LightningDataModule):
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
        self.dataset.tensor = (
            self.dataset.tensor - torch.min(self.dataset.tensor)
        ) / (torch.max(self.dataset.tensor) - torch.min(self.dataset.tensor))
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


"""
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
"""

###############################

if __name__ == "__main__":
    datamodule = HackathonDataModule(datapath, score_path, keypoints, 1)
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

## TODO: vizu
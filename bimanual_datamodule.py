import json
import os
# from isort import file
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from aaha_datamodules import normalize_tensor
from interpolation.tool_box import getPosesBM


class BimanualActionsDataset(Dataset):
    def __init__(self, take_folder: str, gt_file, max_frame: int) -> None:
        """
        Args:
            take_folder (string): Path to the BimanualAction take folder.
            gt_file (string): Path to the ground truth file.
            max_frame (int): Maximum number of frames to consider.
        """
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
                padded_action_point = np.zeros(
                    (self.points.shape[0], max_frame)
                )
                action_point = self.points[
                    :, right_hand_tmstps[i]: right_hand_tmstps[i + 1]
                ]
                padded_action_point[:, : action_point.shape[1]] = action_point
            actions_points.append(padded_action_point)
        for i in range(len(left_hand_tasks)):
            if left_hand_tasks[i] == None:
                frame_to_remove["left_hand"].append(left_hand_tmstps[i])
            else:
                padded_action_point = np.zeros(
                    (self.points.shape[0], max_frame)
                )
                action_point = self.points[
                    :, left_hand_tmstps[i]: left_hand_tmstps[i + 1]
                ]
                padded_action_point[:, : action_point.shape[1]] = action_point
                actions_points.append(padded_action_point)
        self.actions_points = torch.FloatTensor(np.stack(actions_points))

        # Replace None with -1
        right_hand_tasks = [-1 if x == None else x for x in right_hand_tasks]
        left_hand_tasks = [-1 if x == None else x for x in left_hand_tasks]

        self.actions_gt = torch.LongTensor(
            np.concatenate((right_hand_tasks, left_hand_tasks))
        )

    def __len__(self):
        return len(self.actions_points)

    def __getitem__(self, index):
        return self.actions_points[index], self.actions_gt[index]


def get_bmdataset(take_folder, gt_file, max_frame):
    """
    Return a BimanualActionsDataset for a given take    
    Args:
        take_folder (string): Path to the BimanualAction take folder.
        gt_file (string): Path to the ground truth file.
        max_frame (int): Maximum number of frames to consider.
    """
    ds = BimanualActionsDataset(take_folder, gt_file, max_frame)
    return ds


class StackTakes(Dataset):
    """
    Stack all the takes of a dataset
    """
    def __init__(self, takes):
        self.takes = takes
        self.tensor = [x.actions_points for x in self.takes]
        self.gt = [x.actions_gt for x in self.takes]
        self.tensor = torch.cat(self.tensor, dim=0)
        self.gt = torch.cat(self.gt, dim=0)

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, index):
        return self.tensor[index], self.gt[index]


def get_bimanual_actions_dataset(
    max_frame, root_dir="/home/nathan/bmds/"
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
                    print(sub_folder, task_folder, take_folder)
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

    dataset = StackTakes(takes)
    normalize_tensor(dataset.tensor)
    print(dataset.tensor.shape)

    return dataset


class BimanualActionsDataModule(pl.LightningDataModule):
    """
    DataModule for the KIT Bimanual Actions dataset.
    """
    def __init__(self, datapath='/home/nathan/bmds/', batch_size: int = 1, max_frame=49969):
        """
        Args:
            datapath (str): path to the KIT Bimanual Actions dataset
            batch_size (int): size of the batch
            max_frame (int): maximum number of frames to consider (default: AHA video max length)
        """
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.max_frame = max_frame

    def setup(self, stage='whatever') -> None:
        self.dataset = get_bimanual_actions_dataset(
            self.max_frame, self.datapath)
        self.train_ds = self.dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            self.batch_size,
            shuffle=False,
            num_workers=1,
        )

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision.transforms import v2
from torchvision.io import read_video

import pandas as pd

import nibabel as nib
import os
import configparser
from io import StringIO

from pathlib import Path
import random

class MimicEchoDataset(Dataset):

    def __init__(self, data_dir: Path, frames_count=16, apply_consecutive_sampling=False):
        
        self._data_dir = data_dir
        
        self._frames_count = frames_count
        self._apply_consecutive_sampling = apply_consecutive_sampling

        self.filespath = self._load_video_filepaths()
        self.transforms = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((224, 224)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __getitem__(self, idx):

        video_filepath = self.filespath[idx]

        vframes, _, _ = read_video(str(video_filepath.resolve()), pts_unit='sec', output_format='TCHW')
        num_frames = vframes.shape[0]
        # print(vframes.size())

        ### sampling strategy for consecutive frames ###
        if num_frames < self._frames_count:
            indices = torch.linspace(0, num_frames-1, self._frames_count).long()
        else:
            if self._apply_consecutive_sampling is True:
                indices = consecutive_sampling(num_frames, self._frames_count)
            else:
                indices = torch.linspace(0, num_frames-1, self._frames_count).long()
        
        sampled_vframes = vframes[indices]
        pixel_values = self.transforms(sampled_vframes)

        return pixel_values, indices 
    
    def __len__(self):
        return len(self.filespath)

    def _load_video_filepaths(self):
        filepaths = self._data_dir.rglob("*.avi")
        # print(len(list(filepaths)))
        return list(filepaths)

class EchonetDynamic(Dataset):
    def __init__(self, data_dir, frames_count:int=16, mode:str="train", task:str="ef_classification"):
        
        self._data_dir = data_dir
        self._frames_count = frames_count
        self._mode = mode
        self._task = task

        self.filespath = self._load_videos_filepath()
        self.annotation = pd.read_csv(Path(self._data_dir/'FileList.csv'))
        if self._mode == "train":
            self.transforms = v2.Compose([
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(224, 224)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.transforms = v2.Compose([
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ])
    
    def __getitem__(self, idx):
        video_filepath = self.filespath[idx]

        ###========get pixel values========###
        vframes, _, _ = read_video(str(video_filepath.resolve()), pts_unit='sec', output_format='TCHW')
        num_frames = vframes.shape[0]
        indices = torch.linspace(0, num_frames-1, self._frames_count).long()
        sampled_vframes = vframes[indices]
        pixel_values = self.transforms(sampled_vframes)

        ###========get targets for different tasks========###
        row = self.annotation[self.annotation["FileName"] == video_filepath.stem]

        if self._task == "ef_regression":
            return pixel_values, torch.tensor([row["EF"].item()/100.0])
        elif self._task == "esv_regression":
            return pixel_values, torch.tensor([row["ESV"].item()]).log()
        elif self._task == "edv_regression":
            return pixel_values, torch.tensor([row["EDV"].item()]).log()
        elif self._task == "ef_classification":
            ef_value = torch.tensor([row["EF"].item()/100.0])
            ef_label = 0 if ef_value < 0.55 or ef_value > 0.70 else 1
            return pixel_values, torch.tensor(ef_label).float()
        else:
            raise ValueError(f"Argument 'task={self._task}' doesn't exist.")


    def _load_videos_filepath(self):
        files_df = pd.read_csv(Path(self._data_dir)/'FileList.csv')

        files_df['Split'] = files_df['Split'].str.lower()
        split_data = files_df[files_df['Split'] == self._mode]

        videos_filepath = []
        for _, row in split_data.iterrows():
            video_filepath = Path(self._data_dir/'Videos'/(row["FileName"]+'.avi'))
            videos_filepath.append(video_filepath)

        return videos_filepath

    def __len__(self):
        return len(self.filespath)


class Camus(Dataset):
    def __init__(self, data_dir, frames_count:int=16, mode:str="train", task:str="ef_classification"):

        self._data_dir = data_dir
        self._frames_count = frames_count
        self._mode = mode
        self._task = task

        self.patients_id = self._generate_patients_id()[self._mode]
        self.suffix = "_4CH_half_sequence.nii.gz"
        self.annotation_filename = "Info_4CH.cfg"

        if self._mode == "train":
            self.transforms = v2.Compose([
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(224, 224)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.transforms = v2.Compose([
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ])

    def __getitem__(self, idx):

        ###========get video path and its annotation file path========###
        patient_id =self.patients_id[idx]
        video_filepath, annotation_filepath = self._get_video_filepath(patient_id)

        ###========get pixel values========###
        vframes = torch.tensor(nib.load(video_filepath).get_fdata())
        vframes = vframes.permute(2, 0, 1).unsqueeze(1).repeat(1, 3, 1, 1)
        num_frames = vframes.shape[0]
        indices = torch.linspace(0, num_frames - 1, self._frames_count).long()
        sampled_vframes = vframes[indices]
        pixel_values = self.transforms(sampled_vframes)

        ###========get targets for different tasks========###
        config = self._read_cfg_file(annotation_filepath)
        if self._task == "ef_regression":
            return pixel_values, torch.tensor(float(config['TOP']["EF"])/100.)
        elif self._task == "esv_regression":
            return pixel_values, torch.tensor(float(config['TOP']["ES"])).log()
        elif self._task == "edv_regression":
            return pixel_values, torch.tensor(float(config['TOP']["ED"])).log()
        elif self._task == "ef_classification":
            ef_value = torch.tensor(float(config['TOP']["EF"])/100.)
            ef_label = 0 if ef_value < 0.55 or ef_value > 0.70 else 1
            return pixel_values, torch.tensor(ef_label).float()
        else:
            raise ValueError(f"Argument 'task={self._task}' doesn't exist.")


    def __len__(self):
        return len(self.patients_id)


    def _get_video_filepath(self, patient_id:str):
        subdirectory_path = self._data_dir/patient_id
        video_filepath = subdirectory_path/(patient_id+self.suffix)
        annotation_filepath = subdirectory_path/self.annotation_filename
        # print(video_filepath)
        return video_filepath, annotation_filepath

        
    def _generate_patients_id(self):
        patients_id = [f"patient0{i:03}" for i in range(1, 500+1)]
        train_size = 400
        val_size = 50
        test_size = 50

        # random.seed(1)
        # random.shuffle(patients_id)
        train_patients_id = patients_id[:train_size]
        val_patients_id = patients_id[train_size:train_size+val_size]
        test_patients_id = patients_id[-test_size:]

        return {
            "train": train_patients_id,
            "val": val_patients_id,
            "test": test_patients_id,
        }


    def _read_cfg_file(self, path):
        config = configparser.ConfigParser()
        with open(path, 'r') as f:
            config_str = '[TOP]\n' + f.read()
        config_fp = StringIO(config_str)
        config.read_file(config_fp)
        return config

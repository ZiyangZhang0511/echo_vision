import torch
from torch.utils.data import Dataset

from torchvision.transforms import v2
from torchvision.io import read_video

import pandas as pd

from pathlib import Path


class EchonetDynamicDataset(Dataset):
    
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
            return pixel_values, torch.tensor([row["EF"].item()/100.0]), indices
        elif self._task == "esv_regression":
            return pixel_values, torch.tensor([row["ESV"].item()]).log(), indices
        elif self._task == "edv_regression":
            return pixel_values, torch.tensor([row["EDV"].item()]).log(), indices
        elif self._task == "ef_classification":
            ef_value = torch.tensor([row["EF"].item()/100.0])
            ef_label = 0 if ef_value < 0.50 else 1
            return pixel_values, torch.tensor(ef_label).float(), indices
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
import torch
from torch.utils.data import Dataset

from torchvision.io import read_video
from torchvision.transforms import v2

from pathlib import Path


class MimicEchoDataset(Dataset):

    def __init__(self, data_dir: Path, frames_count=16):
        
        self._data_dir = data_dir
        
        self._frames_count = frames_count

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
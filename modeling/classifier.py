from transformers import (
        VideoMAEModel,
        VivitModel, 
        VivitConfig,
)

import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights

from .videomae import get_echo_encoder
from .stff_net import SpatioTemporalFeatureFusionNet

class VideoBinaryClassifier(nn.Module):

    def __init__(self, feat_extractor:str, pretrained_checkpiont_path:str=None, stff_net_flag:bool=False):
        super(VideoBinaryClassifier, self).__init__()
        self._feat_extractor = feat_extractor
        self._stff_net_flag = stff_net_flag

        if self._feat_extractor == "vanilla_videomae":
            self.pretrained_encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
            config = self.pretrained_encoder.config
            hidden_size = config.hidden_size

        elif self._feat_extractor == "vivit":
            config = VivitConfig(num_frames=16)
            self.pretrained_encoder  = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", config=config, ignore_mismatched_sizes=True)
            hidden_size = self.pretrained_encoder.config.hidden_size

        elif self._feat_extractor == "videoresnet":
            self.pretrained_encoder = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            hidden_size = 400
        
        elif self._feat_extractor == "echo_videomae":
            self.pretrained_encoder = get_echo_encoder(pretrained_checkpiont_path)
            hidden_size = self.pretrained_encoder.config.hidden_size
        else:
            ValueError("the value of 'feat_extractor' is wrong!!!")

        if self._stff_net_flag:
            self.stff_net = SpatioTemporalFeatureFusionNet(feat_dim=768, size_feat_map=(8, 14, 14))
            hidden_size = 768 * 2
        
        self.classifier = nn.Sequential(
            self.make_classifer_block(hidden_size, 200),
            self.make_classifer_block(200, 200 // 2),
            self.make_classifer_block(200 // 2, 1, final_layer=True),
        )
    
    def forward(self, pixel_values, temporal_indices):
        
        if self._feat_extractor == "vanilla_videomae":
            video_embeddings = self.pretrained_encoder(pixel_values).last_hidden_state
            video_embeddings, _ = torch.max(video_embeddings, dim=1)

        elif self._feat_extractor == "vivit":
            video_embeddings = self.pretrained_encoder(pixel_values).last_hidden_state[:, 0, :]
        
        elif self._feat_extractor == "videoresnet":
            video_embeddings = self.pretrained_encoder(pixel_values.permute(0, 2, 1, 3, 4).contiguous())

        elif self._feat_extractor == "echo_videomae":
            video_embeddings = self.pretrained_encoder(pixel_values).last_hidden_state
            
            if self._stff_net_flag:
                # print(video_embeddings.size(), temporal_indices.size())
                video_embeddings = self.stff_net(video_embeddings, temporal_indices)
            else:
                video_embeddings, _ = torch.max(video_embeddings, dim=1)
        
        logits = self.classifier(video_embeddings)
        return logits

    def make_classifer_block(self, input_channels, output_channels, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.BatchNorm1d(output_channels),
                nn.ReLU()
            )
        else: # final Layer
            return nn.Sequential(
                nn.Linear(input_channels, output_channels)
            )
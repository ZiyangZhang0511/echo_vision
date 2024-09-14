from transformers import (
        VideoMAEConfig, 
        VideoMAEForPreTraining, 
        VideoMAEModel,
        VivitModel, 
        VivitConfig,
)
import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights

def get_videomae_for_pretraining():
    configuration = VideoMAEConfig()
    configuration.image_size = 224
    model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base", config=configuration)
    return model



class VideoBinaryClassifier(nn.Module):
    def __init__(self, feat_extractor:str):
        super(VideoBinaryClassifier, self).__init__()
        self._feat_extractor = feat_extractor

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

        else:
            ValueError("the value of 'feat_extractor' is wrong!!!")
        
        self.classifier = nn.Sequential(
            self.make_classifer_block(hidden_size, 200),
            self.make_classifer_block(200, 200 // 2),
            self.make_classifer_block(200 // 2, 1, final_layer=True),
        )
    
    def forward(self, pixel_values):
        
        if self._feat_extractor == "vanilla_videomae":
            video_embeddings = self.pretrained_encoder(pixel_values).last_hidden_state
            video_embeddings, _ = torch.max(video_embeddings, dim=1)
        elif self._feat_extractor == "vivit":
            video_embeddings = self.pretrained_encoder(pixel_values).last_hidden_state[:, 0, :]
        elif self._feat_extractor == "videoresnet":
            video_embeddings = self.pretrained_encoder(pixel_values.permute(0, 2, 1, 3, 4).contiguous())
        
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
        

class VideoRegressor(nn.Module):
    def __init__(self, feat_extractor:str):
        super(VideoRegressor, self).__init__()
        self._feat_extractor = feat_extractor

        if self._feat_extractor == "vanilla_videomae":
            self.pretrained_encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
            config = self.pretrained_encoder.config
            hidden_size = config.hidden_size

        elif self._feat_extractor == "vivit":
            config = VivitConfig(num_frames=16)
            self.pretrained_encoder = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", config=config, ignore_mismatched_sizes=True)
            # print(self.pretrained_encoder)

            nn.init.xavier_uniform_(self.pretrained_encoder.pooler.dense.weight)
            nn.init.zeros_(self.pretrained_encoder.pooler.dense.bias)
            
            hidden_size = self.pretrained_encoder.config.hidden_size

        elif self._feat_extractor == "videoresnet":
            self.pretrained_encoder = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            hidden_size = 400

        else:
            ValueError("the value of 'feat_extractor' is wrong!!!")
        
        self.regressor = nn.Sequential(
            self.make_regressor_block(hidden_size, 200),
            self.make_regressor_block(200, 200 // 2),
            self.make_regressor_block(200 // 2, 1, final_layer=True),
        )
    
    def forward(self, pixel_values):
        
        if self._feat_extractor == "vanilla_videomae":
            video_embeddings = self.pretrained_encoder(pixel_values).last_hidden_state
            video_embeddings, _ = torch.max(video_embeddings, dim=1)
        elif self._feat_extractor == "vivit":
            video_embeddings = self.pretrained_encoder(pixel_values).pooler_output
        elif self._feat_extractor == "videoresnet":
            video_embeddings = self.pretrained_encoder(pixel_values.permute(0, 2, 1, 3, 4).contiguous())
        
        logits = self.regressor(video_embeddings)
        return logits

    def make_regressor_block(self, input_channels, output_channels, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.BatchNorm1d(output_channels),
                nn.ReLU()
            )
        else: # Final Layer
            return nn.Sequential(
                nn.Linear(input_channels, output_channels)
            )
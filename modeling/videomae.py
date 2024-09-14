import torch

from transformers import (
        VideoMAEConfig, 
        VideoMAEForPreTraining,
        )

from accelerate import Accelerator


def get_videomae_for_pretraining():
    configuration = VideoMAEConfig()
    configuration.image_size = 224
    model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base", config=configuration)
    return model

def get_echo_encoder(pretrained_checkpiont_path:str):
    
    accelerator = Accelerator()
    configuration = VideoMAEConfig()
    model = VideoMAEForPreTraining.from_pretrained(
        "MCG-NJU/videomae-base",
        config=configuration,
        )
    model = accelerator.prepare(model)
    accelerator.load_state(pretrained_checkpiont_path)
    
    echo_encoder = model.videomae

    return echo_encoder



if __name__ == "__main__":

    accelerator = Accelerator()

    # optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, betas=(0.9, 0.95), weight_decay=0.05)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    # scheduler = LinearWarmupCosineAnnealingLR(
    #     optimizer,
    #     warmup_epochs=10, 
    #     max_epochs=args.num_epochs, 
    #     warmup_start_lr=1e-5, 
    #     eta_min=1e-7,
    # )
    configuration = VideoMAEConfig()
    model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base", config=configuration)

    model = accelerator.prepare(model)
    print(model.videomae.embeddings.patch_embeddings.projection.weight.data.max())
    accelerator.load_state("/home/olg7848/p32335/my_research/echo_vision/checkpoints/checkpoints_pretrain/half/checkpoint_50")
    print(model.videomae.embeddings.patch_embeddings.projection.weight.data.max())





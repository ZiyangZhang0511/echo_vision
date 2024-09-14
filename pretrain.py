import argparse
from pathlib import Path
import time
from tqdm import tqdm
import multiprocessing as mp

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from accelerate import Accelerator

from datasets.mimic import MimicEchoDataset

from modeling.videomae import get_videomae_for_pretraining

PRETRAIN_CHECKPOINTS_DIR = "/projects/p32335/my_research/echo_vision/checkpoints/checkpoints_pretrain/half/"

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
             "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
             "and an Nvidia Ampere GPU.",
    )

    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--initial_lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--num_workers", default=8 , type=int)

    parser.add_argument("--cur_epoch", default=-1, type=int)
    parser.add_argument("--resume_pretraining", action="store_true")

    args = parser.parse_args()

    return args


def train_function(model, dataloader, args):

    frames_count = 16
    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    seq_length = (frames_count // model.config.tubelet_size) * num_patches_per_frame

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, betas=(0.9, 0.95), weight_decay=0.05)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=10, 
        max_epochs=args.num_epochs, 
        warmup_start_lr=1e-5, 
        eta_min=1e-7,
    )

    model, dataloader, optimizer, scheduler = accelerator.prepare(
            model, dataloader, optimizer, scheduler
    )

    if not args.resume_pretraining:
        cur_epoch = -1
        accelerator.save_state(PRETRAIN_CHECKPOINTS_DIR + f"checkpoint_{cur_epoch}")
    else: 
        cur_epoch = args.cur_epoch
        accelerator.load_state(PRETRAIN_CHECKPOINTS_DIR + f"checkpoint_{cur_epoch}")

    accelerator.print("Starting the training model...")
    
    step = 0
    for epoch in range(cur_epoch+1, cur_epoch+1 + args.num_epochs):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(tqdm(dataloader)):
            pixel_values, _ = batch
            pixel_values = pixel_values.to(accelerator.device)
            cur_batch_size = pixel_values.shape[0]
            bool_masked_pos = torch.randint(0, 2, (1, seq_length)).repeat(cur_batch_size, 1).bool()

            outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
            loss = outputs.loss

            epoch_loss += loss.item() / len(dataloader)
            
            accelerator.backward(loss)
            optimizer.step()
            # scheduler.step(epoch+i / len(dataloader))
            optimizer.zero_grad()

            if step % 1000 == 0 and epoch < 5:
                accelerator.print(f"step {step}, current lr: {scheduler.get_lr()}, running loss: {loss.item()}")
            
            step += 1    
            
        scheduler.step()

        accelerator.save_state(PRETRAIN_CHECKPOINTS_DIR + f"checkpoint_{epoch}")
        accelerator.print(f"Epoch {epoch}, current_lr: {scheduler.get_last_lr()}, train Loss: {epoch_loss}")
        accelerator.print(f"saved at epoch {epoch}")
        

def main():
    args = get_parser()

    ###========get pretrained dataset and dataloader========###
    data_dir = Path(args.data_dir)
    mimic_dataset = MimicEchoDataset(data_dir, frames_count=16)
    dataloader = DataLoader(mimic_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        prefetch_factor=2,
        pin_memory=True
    )

    ###========get model========###
    model = get_videomae_for_pretraining()

    ###========train model on mimic dataset and save checkpoint========###
    train_function(model, dataloader, args)



if __name__ == "__main__":
    main()
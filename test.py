import argparse
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from accelerate import Accelerator

from datasets.echonet import EchonetDynamicDataset

from modeling.classifier import VideoBinaryClassifier
from modeling.regressor import VideoRegressor

from utilities import utils

DATAPATH_DICT = {
    "echonet_dynamic": Path("/projects/p32335/Research_on_AI_medicine/EchoGPT/VideoEncoder/Data/EchoNet-Dynamic"),
    "camus": Path("/projects/p32335/Research_on_AI_medicine/EchoGPT/VideoEncoder/Data/CAMUS"),
}

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name", 
        default=None, 
        type=str, 
        choices=["echonet_dynamic", "camus"],                
        help="downstream task dataset to be finetuned.",
    )
    parser.add_argument(
        "--feat_extractor", 
        default="echo_videomae", 
        type=str, 
        choices=["vivit", "videoresnet", "vanilla_videomae", "echo_videomae"],
    )
    parser.add_argument(
        "--task_type", 
        default="ef_regression", 
        type=str, 
        choices=["ef_regression", "esv_regression", "edv_regression", "ef_classification"],
    )

    parser.add_argument(
        "--checkpiont_path", 
        type=str,
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
             "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
             "and an Nvidia Ampere GPU.",
    )

    parser.add_argument("--stff_net_flag", action="store_true")

    parser.add_argument("--num_workers", default=8, type=int)

    args = parser.parse_args()

    return args


def main():

    args = get_parser()

    ###========get test dataset========###
    test_dataset = EchonetDynamicDataset(DATAPATH_DICT[args.dataset_name], mode="test", task=args.task_type)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    ###========get model with weights========###
    if args.task_type == "ef_classification":
        model = VideoBinaryClassifier(args.feat_extractor, stff_net_flag=args.stff_net_flag)
        criterion = nn.BCEWithLogitsLoss()
    elif args.task_type in ["ef_regression", "esv_regression", "edv_regression"]:
        model = VideoRegressor(args.feat_extractor, stff_net_flag=args.stff_net_flag)
        criterion = nn.MSELoss()
        
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    accelerator.load_state(args.checkpiont_path)

    ###========test model for performance========###
    accelerator.print("Starting test model...")
    test_loss, test_metrics_dict = utils.test(accelerator, model, test_dataloader, criterion, "test", args)


if __name__ == "__main__":

    main()
import os
from pathlib import Path

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np

import torch

def compute_metrics(logits, targets, task_type:str, threshold=0.5):

    if task_type == "ef_classification":
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).float()
        predictions = predictions.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions)
        auc = roc_auc_score(targets, probabilities.cpu().detach().numpy())
        
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc,
        }

    elif task_type in ["ef_regression", "esv_regression", "edv_regression"]:
        targets = targets.cpu().detach().numpy()
        logits = logits.cpu().detach().numpy()

        mae = mean_absolute_error(targets, logits)
        mse = mean_squared_error(targets, logits)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, logits)

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }


def get_the_most_recent_dir(directory:str):
    all_dirs = [d for d in Path(directory).iterdir() if d.is_dir()]
    sorted_dirs = sorted(all_dirs, key=lambda x: x.stat().st_mtime)
    return str(sorted_dirs[-1].resolve())


def test(accelerator, model, dataloader, criterion, mode:str, args):
    model.eval()
    total_loss = 0
    total_logits, totoal_targets = [], []
    for pixel_values, targets, temporal_indices in dataloader:
        targets = targets.view(-1, 1).to(accelerator.device)
        
        with torch.no_grad():
            logits = model(pixel_values.to(accelerator.device), temporal_indices.to(accelerator.device))
        
        loss = criterion(logits, targets)

        total_loss += loss.item() / len(dataloader)
        total_logits.append(accelerator.gather(logits))
        totoal_targets.append(accelerator.gather(targets))
    total_logits = torch.cat(total_logits).detach().cpu()
    totoal_targets = torch.cat(totoal_targets).detach().cpu()

    metrics_dict = compute_metrics(total_logits, totoal_targets, args.task_type)
    accelerator.print(f"{mode} loss: {total_loss}, {mode} metrics: {metrics_dict}")

    return total_loss, metrics_dict
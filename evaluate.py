import torch
import torch.nn as nn
import numpy as np
import os
from dataloader import getDataloader
import argparse
import pandas as pd
from loss import WeightedBinaryCrossEntropyLoss
from models import *

def accuracy(outputs, labels):
    """
    Calculate accuracy between predicted outputs and ground truth labels.

    Args:
    - outputs (torch.Tensor): Predicted outputs from the model.
    - labels (torch.Tensor): Ground truth labels.

    Returns:
    - float: Accuracy value.
    """
    correct = torch.sum(torch.eq(outputs, labels).long())
    accuracy = correct / np.prod(np.array(outputs.shape))
    return float(accuracy)

def IOU(outputs, labels):
    """
    Calculate Intersection over Union (IoU) between predicted outputs and ground truth labels.

    Args:
    - outputs (torch.Tensor): Predicted outputs from the model.
    - labels (torch.Tensor): Ground truth labels.

    Returns:
    - float: IoU value.
    """
    intersection = (outputs * labels).sum()
    union = (outputs + labels - (outputs * labels)).sum()

    iou = intersection / union

    return float(iou.mean())


def evaluate(model, validation_loader):
    """
    Evaluate the model on the validation set.

    Args:
    - model (nn.Module): The PyTorch model to be evaluated.
    - validation_loader (DataLoader): DataLoader for validation data.

    Returns:
    - dict: Dictionary containing evaluation metrics (Average Loss, Average Accuracy, Average IoU).
    """
    running_vloss = 0.0
    acc = 0.0
    iou = 0.0
    loss_fn = WeightedBinaryCrossEntropyLoss()
    model.eval()
    count = 0.0

    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
            voutputs = torch.round(torch.sigmoid(voutputs))
            acc += accuracy(voutputs, vlabels) 
            iou += IOU(voutputs, vlabels) 

    avg_vloss = running_vloss / (i + 1)
    avg_acc = acc / (i + 1)
    avg_iou = iou / (i + 1)
    print('LOSS valid {}'.format(avg_vloss))
    print('Average Accuracy valid {}'.format(avg_acc))
    print('Average IOU valid {}'.format(avg_iou))
    return {'Average Loss':[float(avg_vloss)], 'Average Accuracy':[avg_acc], 'Average IOU':[avg_iou]}


def parse_args():
    """
    Parse command line arguments for model evaluation.

    Returns:
    - argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluate sports ball image segmentation model')
    parser.add_argument(
        '--model', type=type, default=UNetMobileNetV2fixed,
        help='Model to train (default: "UNetMobileNetV2fixed")'
    )
    parser.add_argument(
        '--n-epochs', type=int, default=3,
        help='Number of epochs (default: 3)'
    )
    parser.add_argument(
            '--augment', type=bool, default=False,
        help='Whether to augment training data or not (default: False)'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    validation_loader = getDataloader(mode='val')
    model = args.model()
    augment = args.augment
    aug = "_aug" if augment else ""
    model.load_state_dict(torch.load(f'saved models/{model.name}_{args.n_epochs}_epochs{aug}.pt'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    results = evaluate(model, validation_loader)
    df_results = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    df_results.to_csv(f'results/{args.model}_{args.n_epochs}_epochs{aug}.csv')
import torch
import torch.nn as nn
import numpy as np
import models
from dataloader import getDataloader
import argparse
import pandas as pd

def accuracy(outputs, labels):
    correct = torch.sum(torch.eq(outputs, labels).long())
    accuracy = correct / np.prod(np.array(outputs.shape))
    return float(accuracy)

def IOU(outputs, labels):
    intersection = (outputs * labels).sum()
    union = (outputs + labels - (outputs * labels)).sum()

    iou = intersection / union

    return float(iou.mean())

def evaluate(model, validation_loader):
    running_vloss = 0.0
    acc = 0
    batch_size = validation_loader.batch_size
    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
            ACC += accuracy(voutputs, vlabels) / vinputs.shape[0]
            IOU += IOU(voutputs, vlabels) / vinputs.shape[0]

    avg_vloss = running_vloss / (i + 1)
    avg_acc = ACC / (i + 1)
    avg_iou = IOU / (i + 1)
    print('LOSS valid {}'.format(avg_vloss))
    print('Average Accuracy valid {}'.format(avg_acc))
    print('Average IOU valid {}'.format(avg_iou))
    return {'Average Loss':avg_vloss, 'Average Accuracy':avg_acc, 'Average IOU':avg_iou}


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate sports ball image segmentation model')
    parser.add_argument(
        '--model', type=function, default=models.UNetMobileNetV2fixed,
        help='Model to train (default: models.UNetMobileNetV2fixed)'
    )
    parser.add_argument(
        '--n-epochs', type=int, default=3,
        help='Number of epochs (default: 3)'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    validation_loader = getDataloader(mode='val')
    model = args.model()
    model.load_state_dict(torch.load(f'saved models/{args.model}_{args.n_epochs}_epochs.pt'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    results = evaluate(model, validation_loader)
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'results/{args.model}_{args.n_epochs}_epochs.csv') 
import torch
import torch.nn as nn
from models import *
from dataloader import getDataloader
import argparse
import os
from tqdm import tqdm
from loss import WeightedBinaryCrossEntropyLoss

def train_one_epoch(model, training_loader, accumulation_steps = 1):
    """
    Train the model for one epoch using the specified training loader.

    Args:
    - model (nn.Module): The PyTorch model to be trained.
    - training_loader (DataLoader): DataLoader for training data.

    Returns:
    - float: The average training loss for the epoch.
    """
    ##loss_fn = nn.BCEWithLogitsLoss() ## we shouldn't use this loss when working with more than one class
    loss_fn = WeightedBinaryCrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    running_loss = 0.
    last_loss = 0.
    final_loss = 0.

    for i, data in tqdm(enumerate(training_loader)):
        inputs, labels = data
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        if  (i+1)  % accumulation_steps ==  0:  
            optimizer.step()  
            optimizer.zero_grad() 

        running_loss += loss.item()
        final_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100
            print(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0.

    return final_loss / (i+1)

def train_all(model, n_epochs, training_loader):
    """
    Train the model for multiple epochs.

    Args:
    - model (nn.Module): The PyTorch model to be trained.
    - n_epochs (int): Number of epochs for training.
    - training_loader (DataLoader): DataLoader for training data.

    Returns:
    - nn.Module: The trained model.
    """
    for epoch in range(n_epochs):
        print(f'EPOCH {epoch + 1}:')

        model.train(True)
        avg_loss = train_one_epoch(model, training_loader)

        print(f'LOSS train {avg_loss}')
    return model


def parse_args():
    """
    Parse command line arguments.

    Returns:
    - argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train image segmentation model')
    parser.add_argument(
        '--model', type=type, default=UNetMobileNetV2fixed,
        help='Model to train (default: models.UNetMobileNetV2fixed)'
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
    
    model = args.model()
    n_epochs = args.n_epochs
    augment = args.augment
    training_loader = getDataloader(mode='train', augment=augment)
    print('Dataloading over')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Device used :{device}")

    trained_model = train_all(model, n_epochs, training_loader)
    
    os.makedirs('saved models', exist_ok=True)
    torch.save(trained_model.state_dict(), f'saved models/{args.model}_{args.n_epochs}_epochs.pt')
import torch
import torch.nn as nn
import models
from dataloader import getDataloader
import argparse
import os

def train_one_epoch(model, training_loader):
    """
    Train the model for one epoch using the specified training loader.

    Args:
    - model (nn.Module): The PyTorch model to be trained.
    - training_loader (DataLoader): DataLoader for training data.

    Returns:
    - float: The average training loss for the epoch.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 
            print(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0.

    return last_loss

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
    parser = argparse.ArgumentParser(description='Train sports ball image segmentation model')
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

    training_loader = getDataloader(mode='train')
    model = args.model()
    n_epochs = args.n_epochs

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    trained_model = train_all(model, n_epochs, training_loader)
    
    os.makedirs('saved models', exist_ok=True)
    torch.save(trained_model.state_dict(), f'saved models/{args.model}_{args.n_epochs}_epochs.pt')
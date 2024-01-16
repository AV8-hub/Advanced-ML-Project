import torch.nn as nn

def down(in_channels, out_channels):
    """
    Define the downsampling block of a U-Net architecture.

    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.

    Returns:
    - nn.Sequential: Sequential block consisting of max pooling, convolution, batch normalization, and ReLU activation.
    """
    return nn.Sequential(
        nn.MaxPool2d(2),
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def up(in_channels, mid_channels, out_channels):
    """
    Define the upsampling block of a U-Net architecture.

    Parameters:
    - in_channels (int): Number of input channels.
    - mid_channels (int): Number of channels in the middle layer.
    - out_channels (int): Number of output channels.

    Returns:
    - nn.Sequential: Sequential block consisting of upsampling, convolution, batch normalization, and ReLU activation.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

        
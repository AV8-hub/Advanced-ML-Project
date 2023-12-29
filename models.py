import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
from utils import up, down

class UNetMobileNetV2fixed(nn.Module):
    """
    U-Net architecture with a fixed MobileNetV2 encoder for image segmentation.

    Args:
    - num_classes (int): Number of output channels/classes for segmentation. Default is 1.

    Attributes:
    - encoder (nn.Module): MobileNetV2 pretrained feature extractor.
    - encoder_layers (list): List of encoder layers for skip connections.
    - classifier (nn.Sequential): Classifier module for the decoder.
    - upsample (nn.Upsample): Upsampling layer.

    Methods:
    - forward(x): Forward pass through the network
    """
    def __init__(self, num_classes=1):
        super(UNetMobileNetV2fixed, self).__init__()

        self.encoder = models.mobilenet_v2(weights='DEFAULT').features

        ## Steps where we will extract the outputs for skip connections, can be changed
        self.encoder_layers = [
            self.encoder[0:2],
            self.encoder[2:4],
            self.encoder[4:7],
            self.encoder[7:14],
            self.encoder[14:19],
            self.encoder[19:24],
            self.encoder[24:],
        ]

        ## The classifier part can be changed; it probably needs to be more complex when the parameters of the pretrained model are fixed
        self.classifier = nn.Sequential(
            nn.Conv2d(4008, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width).
        """
        # Encoder
        skips = []
        for layer in self.encoder_layers:
            x = layer(x)
            skips.append(x)

        # Decoder
        x = skips[-1]
        for i, skip in enumerate(reversed(skips[:-1])):
            x = torch.cat((x, skip), dim=1)
            if i > 0:
                x = self.upsample(x)

        ## Classifier
        x = self.classifier(x)

        return x

    
class UNetMobileNetV2unfixed(nn.Module):
    """
    U-Net architecture with an unfixed MobileNetV2 encoder for image segmentation.

    Args:
    - num_classes (int): Number of output channels/classes for segmentation. Default is 1.

    Attributes:
    - encoder (nn.Module): MobileNetV2 feature extractor with unfixed parameters.
    - encoder_layers (list): List of encoder layers for skip connections.
    - classifier (nn.Sequential): Classifier module for the decoder.
    - upsample (nn.Upsample): Upsampling layer.

    Methods:
    - forward(x): Forward pass through the network.
    """
    def __init__(self, num_classes=1):
        super(UNetMobileNetV2unfixed, self).__init__()

        self.encoder = models.mobilenet_v2(weights='DEFAULT').features

        ## The MobileNetV2 parameters are not fixed anymore
        for param in self.encoder.parameters():
            param.requires_grad = True

        ## Steps where we will extract the outputs for skip connections, can be changed
        self.encoder_layers = [
            self.encoder[0:2],
            self.encoder[2:4],
            self.encoder[4:7],
            self.encoder[7:14],
            self.encoder[14:19],
            self.encoder[19:24],
            self.encoder[24:],
        ]

        ## The classifier part can be changed; it probably needs to be more complex when the parameters of the pretrained model are fixed
        self.classifier = nn.Sequential(
            nn.Conv2d(4008, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width).
        """
        # Encoder
        skips = []
        for layer in self.encoder_layers:
            x = layer(x)
            skips.append(x)

        # Decoder
        x = skips[-1]
        for i, skip in enumerate(reversed(skips[:-1])):
            x = torch.cat((x, skip), dim=1)
            if i > 0:
                x = self.upsample(x)

        ## Classifier
        x = self.classifier(x)

        return x


class UNetMobileNetV2untrained(nn.Module):
    """
    U-Net architecture with an untrained MobileNetV2 encoder for image segmentation.

    Args:
    - num_classes (int): Number of output channels/classes for segmentation. Default is 1.

    Attributes:
    - encoder (nn.Module): MobileNetV2 feature extractor with untrained parameters.
    - encoder_layers (list): List of encoder layers for skip connections.
    - classifier (nn.Sequential): Classifier module for the decoder.
    - upsample (nn.Upsample): Upsampling layer.

    Methods:
    - forward(x): Forward pass through the network.
    """
    def __init__(self, num_classes=1):
        super(UNetMobileNetV2untrained, self).__init__()

        self.encoder = models.mobilenet_v2(weights=None).features

        ## The MobileNetV2 parameters are not fixed anymore
        for param in self.encoder.parameters():
            param.requires_grad = True

        ## Steps where we will extract the outputs for skip connections, can be changed
        self.encoder_layers = [
            self.encoder[0:2],
            self.encoder[2:4],
            self.encoder[4:7],
            self.encoder[7:14],
            self.encoder[14:19],
            self.encoder[19:24],
            self.encoder[24:],
        ]

        ## The classifier part can be changed; it probably needs to be more complex when the parameters of the pretrained model are fixed
        self.classifier = nn.Sequential(
            nn.Conv2d(4008, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width).
        """
        # Encoder
        skips = []
        for layer in self.encoder_layers:
            x = layer(x)
            skips.append(x)

        # Decoder
        x = skips[-1]
        for i, skip in enumerate(reversed(skips[:-1])):
            x = torch.cat((x, skip), dim=1)
            if i > 0:
                x = self.upsample(x)

        ## Classifier
        x = self.classifier(x)

        return x

   
class CustomUnet(nn.Module):
    """
    Custom U-Net architecture for image segmentation.

    Args:
    - num_channels (int): Number of input channels. Default is 3.
    - num_classes (int): Number of output channels/classes for segmentation. Default is 1.

    Attributes:
    - num_channels (int): Number of input channels.
    - num_classes (int): Number of output channels/classes for segmentation.
    - input_layer (nn.Sequential): Input layer with convolutional and normalization operations.
    - down1, down2, down3, down4 (nn.Sequential): Down-sampling layers.
    - up1, up2, up3, up4 (nn.Sequential): Up-sampling layers.
    - output_layer (nn.Conv2d): Output layer with convolutional operation.

    Methods:
    - forward(x): Forward pass through the network.
    """
    def __init__(self, num_channels=3, num_classes=1):
        super(CustomUnet, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.input_layer = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.up1 = up(1024, 512, 256)
        self.up2 = up(512, 256, 128)
        self.up3 = up(256, 128, 64)
        self.up4 = up(128, 64, 64)

        ## the kernel size is weird, we'll probably change it
        self.output_layer = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width).
        """
        x1 = self.input_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.output_layer(x)

        return logits
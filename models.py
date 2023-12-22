import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
from utils import up, down


class UNetMobileNetV2fixed(nn.Module):
    def __init__(self, num_classes = 1):
        super(UNetMobileNetV2fixed, self).__init__()

    
        self.encoder = models.mobilenet_v2(pretrained=True).features
        
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
        
        ## The classifier part can be changed, it probably needs to be more complex when the the parameters of the pretrained model are fixed
        self.classifier = nn.Sequential(
            nn.Conv2d(320, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        skips = []
        for layer in self.encoder_layers:
            x = layer(x)
            skips.append(x)

        # Decoder
        x = skips[-1]
        for skip in reversed(skips[:-1]):
            x = self.upsample(x)
            x = torch.cat((x, skip), dim=1)
        
        ## Classifier
        x = self.classifier(x)

        return x
    
class UNetMobileNetV2unfixed(nn.Module):
    def __init__(self, num_classes = 1):
        super(UNetMobileNetV2unfixed, self).__init__()

    
        self.encoder = models.mobilenet_v2(pretrained=True).features
        
        ## The MobileNetV2 parameters are not fixed anymore
        for param in self.encoder.encoder.parameters():
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
        
        ## The classifier part can be changed, it probably needs to be more complex when the the parameters of the pretrained model are fixed
        self.classifier = nn.Sequential(
            nn.Conv2d(320, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        skips = []
        for layer in self.encoder_layers:
            x = layer(x)
            skips.append(x)

        # Decoder
        x = skips[-1]
        for skip in reversed(skips[:-1]):
            x = self.upsample(x)
            x = torch.cat((x, skip), dim=1)
        
        ## Classifier
        x = self.classifier(x)

        return x

class UNetMobileNetV2untrained(nn.Module):
    def __init__(self, num_classes = 1):
        super(UNetMobileNetV2untrained, self).__init__()

    
        self.encoder = models.mobilenet_v2(pretrained=False).features
        
        ## The MobileNetV2 parameters are not fixed anymore
        for param in self.encoder.encoder.parameters():
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
        
        ## The classifier part can be changed, it probably needs to be more complex when the the parameters of the pretrained model are fixed
        self.classifier = nn.Sequential(
            nn.Conv2d(320, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        skips = []
        for layer in self.encoder_layers:
            x = layer(x)
            skips.append(x)

        # Decoder
        x = skips[-1]
        for skip in reversed(skips[:-1]):
            x = self.upsample(x)
            x = torch.cat((x, skip), dim=1)
        
        ## Classifier
        x = self.classifier(x)

        return x
   
class CustomUnet(nn.Module):
    def __init__(self, num_channels = 3, num_classes = 1):
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
        self.up3 = up(128, 64, 64)
        
        ## the kernel size is weird, we'll probably change it
        self.output_layer = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        
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
    
        
        
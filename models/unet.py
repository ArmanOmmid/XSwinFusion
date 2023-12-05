
import numpy as np
import torch
import torch.nn as nn

from ._network import _Network
from .modules.normal.convolution_triplet import ConvolutionTriplet

class UNet(_Network):
    def __init__(self, channels=[64, 128, 256, 512], weights=None):
        super().__init__()

        assert len(channels) == 4, "UNet must 4 encoder layer channels defined"

        layer_channels_1 = channels[0]
        layer_channels_2 = channels[1]
        layer_channels_3 = channels[2]
        layer_channels_4 = channels[3]

        self.encoder_1 = ConvolutionTriplet(3, layer_channels_1, kernel_size=3) # 160, 320
        self.encoder_2 = ConvolutionTriplet(layer_channels_1, layer_channels_2, kernel_size=3) # 80, 160
        self.encoder_3 = ConvolutionTriplet(layer_channels_2, layer_channels_3, kernel_size=3) # 40, 80
        self.encoder_4 = ConvolutionTriplet(layer_channels_3, layer_channels_4, kernel_size=3) # 20, 40

        self.maxpool = nn.MaxPool2d((2,2))

        self.upconv_3 = nn.ConvTranspose2d(layer_channels_4, layer_channels_3, kernel_size=2, stride=2) # 40, 80
        self.decoder_3 = ConvolutionTriplet(layer_channels_3*2, layer_channels_3, kernel_size=3) # 40, 80

        self.upconv_2 = nn.ConvTranspose2d(layer_channels_3, layer_channels_2, kernel_size=2, stride=2) # 80, 160
        self.decoder_2 = ConvolutionTriplet(layer_channels_2*2, layer_channels_2, kernel_size=3) # 80, 160

        self.upconv_1 = nn.ConvTranspose2d(layer_channels_2, layer_channels_1, kernel_size=2, stride=2) # 160, 320
        self.decoder_1 = ConvolutionTriplet(layer_channels_1*2, layer_channels_1, kernel_size=3) # 80, 160

        self.segmentation_head = nn.Conv2d(layer_channels_1, 1, kernel_size=1) # 1x1 conv

        self.load(weights)

    def forward(self, x: torch.Tensor):
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(self.maxpool(x1))
        x3 = self.encoder_3(self.maxpool(x2))
        x4 = self.encoder_4(self.maxpool(x3))

        y3 = self.upconv_3(x4)
        y3 = torch.cat((x3, y3), dim=1)
        y3 = self.decoder_3(y3)

        y2 = self.upconv_2(y3)
        y2 = torch.cat((x2, y2), dim=1)
        y2 = self.decoder_2(y2)

        y1 = self.upconv_1(y2)
        y1 = torch.cat((x1, y1), dim=1)
        y1 = self.decoder_1(y1)

        y = self.segmentation_head(y1)
        y = y.squeeze(1)
        return y

#############################################################
# Unet0
#############################################################

import os
from abc import ABC
class AbstractNet(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def load(self, weights_path=None, map_location='cpu'):
        if weights_path:
            self.load_state_dict(torch.load(weights_path, map_location=torch.device(map_location)))

    def save(self, weights_path):
        if not os.path.exists(os.path.dirname(weights_path)): 
            os.makedirs(os.path.dirname(weights_path))
        torch.save(self.state_dict(), weights_path)

class UniformTripletLayer(nn.Module):
    def __init__(self, in_channels, layer_channels, kernel_size=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=layer_channels, kernel_size = kernel_size, stride = 1, padding = (kernel_size - 1)//2),
            nn.BatchNorm2d(layer_channels),
            nn.LeakyReLU(),
            nn.Conv2d(layer_channels, out_channels=layer_channels, kernel_size = kernel_size, stride = 1, padding = (kernel_size - 1)//2),
            nn.BatchNorm2d(layer_channels),
            nn.LeakyReLU(),
            nn.Conv2d(layer_channels, out_channels=layer_channels, kernel_size = kernel_size, stride = 1, padding = (kernel_size - 1)//2),
            nn.BatchNorm2d(layer_channels),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.layers(x)

class UNet0(AbstractNet):
    def __init__(self, channels=[64, 128, 256, 512], weights=None, device='cpu'):
        super().__init__()

        assert len(channels) == 4, "UNet must 4 encoder layer channels defined"

        layer_channels_1 = channels[0]
        layer_channels_2 = channels[1]
        layer_channels_3 = channels[2]
        layer_channels_4 = channels[3]

        self.encoder_1 = UniformTripletLayer(3, layer_channels_1, kernel_size=3) # 160, 320
        self.encoder_2 = UniformTripletLayer(layer_channels_1, layer_channels_2, kernel_size=3) # 80, 160
        self.encoder_3 = UniformTripletLayer(layer_channels_2, layer_channels_3, kernel_size=3) # 40, 80
        self.encoder_4 = UniformTripletLayer(layer_channels_3, layer_channels_4, kernel_size=3) # 20, 40

        self.maxpool = nn.MaxPool2d((2,2))

        self.upconv_3 = nn.ConvTranspose2d(layer_channels_4, layer_channels_3, kernel_size=2, stride=2) # 40, 80
        self.decoder_3 = UniformTripletLayer(layer_channels_3*2, layer_channels_3, kernel_size=3) # 40, 80

        self.upconv_2 = nn.ConvTranspose2d(layer_channels_3, layer_channels_2, kernel_size=2, stride=2) # 80, 160
        self.decoder_2 = UniformTripletLayer(layer_channels_2*2, layer_channels_2, kernel_size=3) # 80, 160

        self.upconv_1 = nn.ConvTranspose2d(layer_channels_2, layer_channels_1, kernel_size=2, stride=2) # 160, 320
        self.decoder_1 = UniformTripletLayer(layer_channels_1*2, layer_channels_1, kernel_size=3) # 80, 160

        self.segmentation_head = nn.Conv2d(layer_channels_1, 1, kernel_size=1) # 1x1 conv

        self.load(weights, map_location=device)
        self.to(device)

    def forward(self, x):
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(self.maxpool(x1))
        x3 = self.encoder_3(self.maxpool(x2))
        x4 = self.encoder_4(self.maxpool(x3))

        y3 = self.upconv_3(x4)
        y3 = torch.cat((x3, y3), dim=1)
        y3 = self.decoder_3(y3)

        y2 = self.upconv_2(y3)
        y2 = torch.cat((x2, y2), dim=1)
        y2 = self.decoder_2(y2)

        y1 = self.upconv_1(y2)
        y1 = torch.cat((x1, y1), dim=1)
        y1 = self.decoder_1(y1)

        y = self.segmentation_head(y1)
        y = y.squeeze(1)
        return y

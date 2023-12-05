
import torch.nn as nn

from ...modules import Modulator

class ConvolutionTriplet_Modulated(nn.Module):
    def __init__(self, in_channels, layer_channels, mod_dims, kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList([
            Modulator(mod_dims, in_channels, n_unsqueeze=2, channel_last=False),
            nn.Conv2d(in_channels, out_channels=layer_channels, kernel_size = kernel_size, stride = 1, padding = (kernel_size - 1)//2),
            nn.BatchNorm2d(layer_channels),
            nn.LeakyReLU(),
            Modulator(mod_dims, layer_channels, n_unsqueeze=2, channel_last=False),
            nn.Conv2d(layer_channels, out_channels=layer_channels, kernel_size = kernel_size, stride = 1, padding = (kernel_size - 1)//2),
            nn.BatchNorm2d(layer_channels),
            nn.LeakyReLU(),
            Modulator(mod_dims, layer_channels, n_unsqueeze=2, channel_last=False),
            nn.Conv2d(layer_channels, out_channels=layer_channels, kernel_size = kernel_size, stride = 1, padding = (kernel_size - 1)//2),
            nn.BatchNorm2d(layer_channels),
            nn.LeakyReLU(),
        ])
    def forward(self, x, c):
        for layer in self.layers:
            if isinstance(layer, Modulator):
                x = layer(x, c)
            else:
                x = layer(x)
        return x


import torch.nn as nn

class ConvolutionTriplet(nn.Module):
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

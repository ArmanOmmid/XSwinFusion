
import torch
import torch.nn as nn

from ...modules import Modulator

class PointwiseConvolution_Modulated(nn.Module):
    def __init__(self, in_channels, out_channels, mod_dims, channel_last=True):
        super().__init__()
        self.mod = Modulator(mod_dims, in_channels, n_unsqueeze=2, channel_last=channel_last)
        if channel_last:
            self.pointwise = nn.Linear(in_channels, out_channels)
        else:
            self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, c):
        return self.pointwise(self.mod(x, c))

from typing import *
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import Permute

import torchvision.models.segmentation as segmentation

import kornia.geometry.conversions as conversions

from ._network import _Network
from .xswin import XNetSwinTransformer
from .pointnet import PointNet

from .modules import LambdaModule, ViTEncoderBlock

default_xswin_kwargs = {
    "patch_size" : [8, 8],
    "embed_dim" : 64,
    "depths" : [2, 2, 2],
    "num_heads" : [4, 8, 16],
    "window_size" : [4, 4],
    "mlp_ratio" : 4.0,
    "num_classes" : None,
    "global_stages" : 1,
    "input_size" : None,
    "final_downsample" : False,
    "residual_cross_attention" : True,
    "smooth_conv" : True,
}

class XSwinFusion(_Network):
    def __init__(self, feature_dims=64, resize=None, quaternion=True, 
                 xswin_kwargs=default_xswin_kwargs, pretrained_resnet=False, weights=None, **kwargs):
        super().__init__(**kwargs)

        self.quaternion = quaternion

        self.pretrained = pretrained_resnet
        if pretrained_resnet:
            self.segment_net = segmentation.fcn_resnet50(pretrained=True)
            self.segment_net.classifier[4] = nn.Conv2d(512, feature_dims, kernel_size=(1,1))
            if self.segment_net.aux_classifier is not None:
                self.segment_net.aux_classifier[4] = nn.Conv2d(256, feature_dims, kernel_size=(1,1))
        else:
            xswin_kwargs["num_classes"] = feature_dims
            xswin_kwargs["input_size"] = resize
            self.segment_net = XNetSwinTransformer(**xswin_kwargs)
        
        self.point_net = PointNet(out_channels=feature_dims, embed_dim=feature_dims)

        self.global_net = nn.Sequential(
            nn.Conv1d(feature_dims*2, feature_dims, 1),
            nn.BatchNorm1d(feature_dims),
            nn.LeakyReLU(),
            nn.Conv1d(feature_dims, feature_dims, 1),
            nn.BatchNorm1d(feature_dims),
            nn.LeakyReLU(),
            LambdaModule(lambda x: torch.mean(x, 2, keepdim=True)) # Average Pooling
        )

        self.mix = nn.Sequential(
            nn.Conv1d(feature_dims*3, feature_dims, 1),
            nn.BatchNorm1d(feature_dims),
            nn.LeakyReLU(),
            Permute([0, 2, 1]), # B L C 
            ViTEncoderBlock(num_heads=feature_dims//16, hidden_dim=feature_dims, mlp_dim=feature_dims,
                            dropout=0.0, attention_dropout=0.0,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6)),
            ViTEncoderBlock(num_heads=feature_dims//8, hidden_dim=feature_dims, mlp_dim=feature_dims,
                            dropout=0.0, attention_dropout=0.0,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6)),
            ViTEncoderBlock(num_heads=feature_dims//4, hidden_dim=feature_dims, mlp_dim=feature_dims,
                            dropout=0.0, attention_dropout=0.0,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6)),
            Permute([0, 2, 1]), # B C L
            nn.Conv1d(feature_dims, feature_dims, 1),
            nn.BatchNorm1d(feature_dims),
            nn.LeakyReLU(),
        )

        representation = [
                nn.Conv1d(16, 8, 1),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(),
                nn.Conv1d(8, 4, 1),
             ] if self.quaternion else [nn.Conv1d(16, 9, 1)]

        self.rotation = nn.Sequential(
            nn.Conv1d(feature_dims, 32, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 16, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(16, 16, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            *representation,
            # nn.Tanh(), # SO3 and quaternions are all [-1, 1] components
            Permute([0, 2, 1]), # B C L -> B L C
        )

        self.translation = nn.Sequential(
            nn.Conv1d(feature_dims, 32, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 16, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(16, 8, 1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Conv1d(8, 3, 1),
            Permute([0, 2, 1]), # B C L -> B L C
        )

        self.confidence = nn.Sequential(
            nn.Conv1d(feature_dims, 32, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 16, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(16, 8, 1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Conv1d(8, 1, 1),
            Permute([0, 2, 1]), # B C L -> B L C
        )

        self.load(weights)

    def forward(self, pcd, rgb, mask_indices):

        batch_indices = np.arange(pcd.size(0))[:, None]
        
        rgb = self.segment_net(rgb)
        if self.pretrained:
            rgb = rgb["out"]
        pcd, transforms = self.point_net(pcd)
        rgb = torch.permute(rgb, (0, 2, 3, 1)) # B C H W -> B H W C (channel last for masking)

        row_indices = mask_indices[:, 0, :]
        col_indices = mask_indices[:, 1, :]

        rgb = rgb[batch_indices, row_indices, col_indices] # RGB POINT CLOUD
        rgb = torch.permute(rgb, (0, 2, 1)) # B, L, C -> B, C, L

        x = torch.cat((rgb, pcd), dim=-2) # concat along channel dim (B, C, L)

        g = self.global_net(x)
        g = g.repeat(1, 1, x.shape[-1]) # Broadcasted

        x = torch.cat((x, g), dim=-2) # B, C1 + C2, L

        x = self.mix(x)

        B = np.arange(pcd.size(0))
        R = self.rotation(x)
        T = self.translation(x)
        C = self.confidence(x)
        C = C.squeeze(-1).argmax(dim=-1)

        R = R[B, C, :]
        if self.quaternion:
            R = conversions.quaternion_to_rotation_matrix(R) # It's automatically normalized
        else:
            R = R.view(-1, 3, 3)

        T = T[B, C, :].unsqueeze(-1)

        x = torch.cat((R, T), dim=-1)

        return x, transforms

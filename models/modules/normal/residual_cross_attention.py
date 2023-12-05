from typing import *
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

def _unfold_padding_prep(x: Tensor, window_height: int, window_width: int):

    *B, H, W, C = x.shape

    mod_height = window_height - (H % window_height)
    mod_width = window_width - (W % window_width)

    pad_h1 = mod_height // 2
    pad_h2 = mod_height - pad_h1
    pad_w1 = mod_width // 2
    pad_w2 = mod_width - pad_w1

    if mod_height > 0 or mod_width > 0:
        x = x.permute(0, -1, -3, -2)
        x = F.pad(x, (pad_w1, pad_w2, pad_h1, pad_h2), 'constant', 0)
        x = x.permute(0, -2, -1, -3)

    padding_info = (pad_h1, pad_h2, pad_w1, pad_w2)
    return x, padding_info
    
def _fold_unpadding_prep(x: Tensor, padding_info):

    pad_h1, pad_h2, pad_w1, pad_w2 = padding_info

    if np.sum(padding_info) == 0:
        return x

    # Set 0 dimensional pads to None so they can be ignored with slicing
    padding_info = [None if p == 0 else p for p in padding_info]

    x = x[:, pad_h1:-pad_h2, pad_w1:-pad_w2, :]

    return x

def _extract_windows(feature_map: Tensor, window_height, window_width):
    """
    Extract local windows from a feature map for non-square windows and flatten them.

    Parameters:
    - feature_map: the input feature map, shape [B*, H, W, C]
    - window_height: the height of the window
    - window_width: the width of the window

    Returns:
    - windows: the local windows ready for attention, shape [B*, Window, S, C]
    """
    
    *B, H, W, C = feature_map.shape

    feature_map, padding_info = _unfold_padding_prep(feature_map, window_height, window_width)

    # Unfold the feature map into windows for both dimensions
    unfolded_height = feature_map.unfold(-3, window_height, window_height) # New dim (from H) created at the end

    unfolded_both = unfolded_height.unfold(-3, window_width, window_width) # So we do -3 again for W
    # Send C to the back
    windows = unfolded_both.permute(0, -5, -4, -2, -1, -3)

    # The shape after unfold will be [B*, H_unfolded, W_unfolded, window_height, window_width, C]
    *B, H_unfolded, W_unfolded, H_window, W_window, C = windows.shape
    # We need [*B, H_unfolded, W_unfolded, <WindowContext>, C]
    windows = windows.reshape(*B, H_unfolded, W_unfolded, window_height * window_width, C)

    return windows, padding_info

class SwinResidualCrossAttention(nn.Module):
    def __init__(self, 
                 window_size,
                 embed_dim, 
                 num_heads,
                 attention_dropout = 0.0, 
                 norm_layer = partial(nn.LayerNorm, eps=1e-5),
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert len(window_size) == 2

        self.window_height, self.window_width = window_size

        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.norm = norm_layer(embed_dim)

    def forward(self, x: Tensor, residual: Tensor):

        # Unfolding
        x, _ = _extract_windows(x, self.window_height, self.window_width)
        residual, padding_info = _extract_windows(residual, self.window_height, self.window_width)

        assert x.shape == residual.shape, f"{x.shape} != {residual.shape}"

        *B, H, W, WINDOW, C = residual.shape

        Q = x.reshape(-1, WINDOW, C)
        K = V = residual.reshape(-1, WINDOW, C)

        # output = attended residuals
        output, attention_weights = self.cross_attention(Q, K, V)

        # Folding
        output = output.reshape(*B, H, W, self.window_height, self.window_width, C)
        output = output.permute(0, -5, -3, -4, -2, -1)
        output = output.reshape(*B, H * self.window_height, W * self.window_width, C)

        # Unpadding
        output = _fold_unpadding_prep(output, padding_info)

        output = self.norm(output)

        return output

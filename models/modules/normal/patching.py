
from functools import partial
import torch.nn as nn
from torch import Tensor
from torchvision.ops.misc import MLP, Permute
import torch.nn.functional as F

class Patching(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 patch_size,
                 norm_layer = partial(nn.LayerNorm, eps=1e-5)):
        super().__init__()

        self.patching = nn.Sequential(
            nn.Conv2d(
                embed_dim, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
            ),
            Permute([0, 2, 3, 1]), # B C H W -> B H W C
            norm_layer(embed_dim),
        )

    def forward(self, x):
        return self.patching(x)

class UnPatching(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 patch_size,
                 norm_layer = partial(nn.BatchNorm2d, eps=1e-5)):
        super().__init__()

        self.unpatching = nn.Sequential(
            Permute([0, 3, 1, 2]), # B H W C -> B C H W
            nn.ConvTranspose2d(
                embed_dim, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
            ),
            norm_layer(embed_dim), # NOTE : Swapped out from LayerNorm because we are Conv-ing
        )

    def forward(self, x, target_shape=None):

        x = self.unpatching(x)
        
        if target_shape is not None:
            assert len(target_shape) == 2, "Target Shape be of Length 2: (H, W)"
            H, W = target_shape
            x = _match_dims_padding(x, H, W)

        return x

def _match_dims_padding(x: Tensor, H: int, W: int):

    # We are in Conv Dims here, H, W are last
    mod_height = H - x.size(-2)
    mod_width = W - x.size(-1)

    pad_h1 = mod_height // 2
    pad_h2 = mod_height - pad_h1
    pad_w1 = mod_width // 2
    pad_w2 = mod_width - pad_w1

    if mod_height > 0 or mod_width > 0:
        x = F.pad(x, (pad_w1, pad_w2, pad_h1, pad_h2), 'constant', 0)

    return x

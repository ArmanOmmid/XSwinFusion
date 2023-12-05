import torch
import torch.nn as nn
import numpy as np
import math

from torch import Tensor
from torchvision.ops.misc import MLP, Permute

class Modulator(nn.Module):
    def __init__(self, in_dims, out_dims, n_unsqueeze, gate=False, channel_last=True, *args, **kwargs) -> None:
        """
        in_dims : The dimensions of the modulation embeddings
        out_dims : The dimensions to match the current data (x) dimensions
        n_unsqueeze : Then number of dimensions between BATCH and CHANNELS after permuatations:
            - B H W C  -> 2
            - B C H W  -> 2 (after permutation, we have B H W C)
            - B L C    -> 1
            We need this information for unsqueezing our modulation embeddings
        channel_list : Whether the channel dimension is last or suspected to be in the 3rd to last spot (Conv Permutation)
            - B H W C  -> True
            - B C H W  -> False
        """
        super().__init__(*args, **kwargs)

        self.n_unsqueeze = n_unsqueeze
        self.gate = gate
        self.chunks = 3 if gate else 2

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dims, self.chunks * out_dims, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # For Conv Layers, channels are in the 3rd to last position!
        self.channel_last = channel_last
        if not channel_last:
            self.permute_in = Permute([0, 2, 3, 1])
            self.permute_out = Permute([0, 3, 1, 2])

    def _get_modulation(self, c):
        return self.adaLN_modulation(c).chunk(self.chunks, dim=1)

    def _unsqueeze(self, x):
        if self.n_unsqueeze == 1:
            return x.unsqueeze(1)
        elif self.n_unsqueeze == 2:
            return x.unsqueeze(1).unsqueeze(1)
        else:
            raise NotImplementedError("Should Not Unsqueeze outside of 1 or 2 times ")

    def _modulate(self, x, shift, scale):
        return x * (1 + self._unsqueeze(scale)) + self._unsqueeze(shift)
    
    def forward(self, x: Tensor, c: Tensor, apply_gate=True):

        if not self.channel_last:
            x = self.permute_in(x)

        if not self.gate:
            shift, scale = self._get_modulation(c)

            x = self._modulate(x, shift, scale)
            if not self.channel_last:
                x = self.permute_out(x)
            return x
        else:
            shift, scale, gate = self._get_modulation(c)
            gate = self._unsqueeze(gate)
            x = self._modulate(x, shift, scale)
            if apply_gate:
                x = x * gate
                if not self.channel_last:
                    x = self.permute_out(x)
                return x
            else:
                if not self.channel_last:
                    x = self.permute_out(x)
                    gate = self.permute_out(gate)
                    return x, gate
                return x, gate

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

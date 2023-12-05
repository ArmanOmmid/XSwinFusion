
import torch
import torch.nn as nn

from .embeddings import Modulator, TimestepEmbedder, LabelEmbedder

def initialize_weights(model):
    # Initialize transformer layers:
    def _basic_init(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.trunc_normal_(module.bias, std=1e-6) # nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.trunc_normal_(module.bias, std=1e-6) # nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.trunc_normal_(module.bias, std=1e-6) # nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    model.apply(_basic_init)

def initalize_diffusion(model):
    
    def _modulator_init(module):
        if isinstance(module, Modulator):
            nn.init.constant_(module.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(module.adaLN_modulation[-1].bias, 0)
        elif isinstance(module, TimestepEmbedder):
            nn.init.normal_(module.mlp[0].weight, std=0.02)
            nn.init.normal_(module.mlp[2].weight, std=0.02)
        elif isinstance(module, LabelEmbedder):
            nn.init.normal_(module.embedding_table.weight, std=0.02)
    model.apply(_modulator_init)

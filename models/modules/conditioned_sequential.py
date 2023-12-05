
import torch
import torch.nn as nn

class ConditionedSequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.sequential = nn.ModuleList(modules)

    def forward(self, x, c):
        for module in self.sequential:
            x = module(x, c)
        return x

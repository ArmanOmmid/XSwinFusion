import torch
import torch.nn as nn

class LambdaModule(nn.Module):
    def __init__(self, function, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.function = function

    def forward(self, *args):
        return self.function(*args)

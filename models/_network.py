
from abc import ABC

import os
import torch
import torch.nn as nn

WEIGHTS_EXTENSION = ".pth"

class _Network(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, load_path: str=None, map_location: str="cpu", unexpected_ok=False, missing_ok=False):
        """
        Load weights if weights were specified
        """
        if not load_path: return
        load_path = load_path.split(".")[0] + WEIGHTS_EXTENSION

        strict = not (unexpected_ok or missing_ok)
        loaded_state_dict = torch.load(load_path, map_location=torch.device(map_location))
        if not strict:
            model_state_dict = self.state_dict()

            unexpected_keys = [k for k in loaded_state_dict.keys() if k not in model_state_dict]
            missing_keys = [k for k in model_state_dict.keys() if k not in loaded_state_dict]

            if len(unexpected_keys) > 0:
                if unexpected_ok:
                    print("Unexpected Keys Permitted")
                    print(unexpected_keys)
                else:
                    print("Unexpected Keys Error")
                    print(unexpected_keys)
                    raise KeyError()
            if len(missing_keys) > 0:
                if missing_ok:
                    print("Missing Keys Permitted")
                    print(missing_keys)
                else:
                    print("Missing Keys Error")
                    print(missing_keys)
                    raise KeyError()

        self.load_state_dict(loaded_state_dict, strict=strict)

    def save(self, save_path: str):
        """
        All saves should be under the same path folder, under different tag folders, with the same filename
        """
        save_path = save_path.split(".")[0] + WEIGHTS_EXTENSION
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(self.state_dict(), save_path)

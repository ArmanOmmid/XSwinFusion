import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from abc import ABC
from datetime import datetime

DIR_PATH = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(DIR_PATH, "npz", "train.npz")
TEST_PATH = os.path.join(DIR_PATH, "npz", "test.npz")

class EdgeDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, eager_transform=True, input_transform=None, target_transform=None):
        
        if train:
            npz = np.load(TRAIN_PATH)
        else:
            npz = np.load(TEST_PATH)

        self.raw = npz
        self.eager_transform = eager_transform
        
        inputs = npz["images"]  # array with shape (N,Width,Height,3)
        targets = npz["edges"]  # array with shape (N,Width,Height)

        if not self.eager_transform:
            self.inputs = inputs
            self.targets = targets
        else:
            self.inputs = []
            self.targets = []
            for input, target in zip(inputs, targets):
                self.inputs.append(self.base_input_transform(input))
                self.targets.append(self.base_target_transform(target))
            self.inputs = torch.stack(self.inputs)
            self.targets = torch.stack(self.targets)

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx] 
        target = self.targets[idx]

        if not self.eager_transform:
            input = self.base_input_transform(input)
            target = self.base_target_transform(target)

        if self.input_transform: 
            input = self.input_transform(input)
        if self.target_transform: 
            target = self.target_transform(target)

        return input, target
    
    def base_input_transform(self, x):
        x = torch.from_numpy(x)
        x = x.float() / 255
        x = x.permute(2, 0, 1)
        return x
    
    def base_target_transform(self, x):
        x = torch.from_numpy(x)
        x = x.float() / 255
        return x

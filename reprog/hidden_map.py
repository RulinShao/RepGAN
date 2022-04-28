import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HiddenMap(nn.Module):
    """MLP block"""
    def __init__(self, dim_in, dim_out=None):
        super(HiddenMap, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.main(x)

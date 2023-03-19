import torch
import torch.nn as nn


class IdentifyModel(nn.Module):
    def __init__(self, out_channels=128):
        super(IdentifyModel, self).__init__()
        self.out_channels = out_channels

    def forward(self, input, **kwargs):
        return input
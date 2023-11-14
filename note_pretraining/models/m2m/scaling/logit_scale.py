import torch
import numpy as np
from torch import nn
from .clamp import Clamp

class LogitScale(nn.Module):

    def __init__(self, initial_value=None):
        super().__init__()
        if initial_value is None:
            self._logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self._logit_scale = nn.Parameter(torch.ones([]) * initial_value)
        self._clamp = Clamp()
        self._clamp = self._clamp.apply

    def forward(self, x):
        return x * self._clamp(self._logit_scale).exp()

    def get_value(self):
        return self._logit_scale.detach()

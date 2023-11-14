import torch
import numpy as np
from torch import nn

class UncertaintyWeighting(nn.Module):

    def __init__(self, num_tasks):
        super().__init__()
        self._task_weights = [nn.Parameter(torch.ones([]) * np.log(1)) for _ in range(num_tasks)]

    def forward(self, *losses):
        return torch.sum(
            torch.stack([torch.exp(-weight) * loss + weight for loss, weight in zip(losses, self._task_weights)])
        )

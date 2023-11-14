import torch

class CustomWeighting(object):

    def __init__(self, task_weights):
        super().__init__()
        self._task_weights = task_weights

    def __call__(self, *losses):
        return torch.sum(torch.stack([loss * weight for loss, weight in zip(losses, self._task_weights)]))
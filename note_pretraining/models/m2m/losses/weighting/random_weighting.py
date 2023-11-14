import torch

class RandomWeighting(object):

    def __init__(self, num_tasks):
        super().__init__()
        self._num_tasks = num_tasks

    def __call__(self, *losses):
        weights = torch.softmax(torch.randn(self._num_tasks), dim=-1)
        return torch.sum(torch.stack([(weights[index] * loss) for index, loss in enumerate(losses)]))

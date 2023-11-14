import numpy as np
from torch.autograd import Function

class Clamp(Function):

    @staticmethod
    def forward(ctx, i):
        return i.clamp(0., np.log(100))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
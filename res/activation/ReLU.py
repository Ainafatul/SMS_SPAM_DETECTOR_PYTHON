import numpy as np

from res.activation.Activation import Activation


class ReLU(Activation):

    def forward(self, x):
        self.logit = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * (self.logit > 0)

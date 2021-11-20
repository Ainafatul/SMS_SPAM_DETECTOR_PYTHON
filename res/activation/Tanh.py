import numpy as np

from res.activation.Activation import Activation


class Tanh(Activation):

    def forward(self, x):
        self.logit = x
        return np.tanh(x)

    def backward(self, grad):
        return grad * (1 - self.forward(self.logit) ** 2)

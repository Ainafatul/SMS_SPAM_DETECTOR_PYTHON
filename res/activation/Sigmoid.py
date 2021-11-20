import numpy as np

from res.activation.Activation import Activation


class Sigmoid(Activation):

    def forward(self, x):
        self.logit = x
        return 1 / (1 + np.exp(-x))

    def backward(self, grad):
        return grad * self.forward(self.logit) * (1 - self.forward(self.logit))

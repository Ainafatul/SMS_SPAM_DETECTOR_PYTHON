import numpy as np

from res.activation.Activation import Activation


class Linear(Activation):

    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)

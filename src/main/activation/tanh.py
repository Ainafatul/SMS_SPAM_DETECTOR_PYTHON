import numpy as np

from activation import Activation


class Tanh(Activation):
    def forward_activation(self, x):
        return np.tanh(x)

    def backward_activation(self, x):
        return (1 - np.tanh(x) ** 2)

import numpy as np

from activation import Activation


class Linear(Activation):
    def forward_activation(self, x):
        return x

    def backward_activation(self, x):
        return np.ones_like(x)

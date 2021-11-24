import numpy as np

from activation import Activation


class Sigmoid(Activation):
    def forward_activation(self, x):
        return 1 / (1 + np.nan_to_num(np.exp(-x)))

    def backward_activation(self, x):
        a = self.forward_activation(x)
        return a * (1 - a)

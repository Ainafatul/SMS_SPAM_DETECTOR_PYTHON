import numpy as np

from activation import Activation


class Sigmoid(Activation):
    def forward_activation(self, x):
        # x = np.float64(x)
        return 1 / (1 + np.exp(-x))

    def backward_activation(self, x):
        a = self.forward_activation(x)
        return a * (1 - a)

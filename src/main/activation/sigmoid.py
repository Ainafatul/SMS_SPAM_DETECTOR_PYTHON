import numpy as np

from activation import Activation


class Sigmoid(Activation):
    def forward_activation(self, x):
        return 1 / (1 + np.exp(-x))

    def backward_activation(self, x):
        return self.forward_activation(x) * (1 - self.forward_activation(x))

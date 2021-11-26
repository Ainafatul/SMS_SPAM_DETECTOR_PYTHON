import numpy as np

from activation import Activation


class ReLU(Activation):

    def forward_activation(self, x):
        return np.where(x > 0, x, 0)

    def backward_activation(self, x):
        return np.where(x > 0, 1, 0)


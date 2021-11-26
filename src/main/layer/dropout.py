import numpy as np


class Dropout:

    def __init__(self, p):
        self.p = p

    def forward(self, x, training=False):
        if training:
            self.mask = np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
            return x * self.mask
        else:
            return x

    def backward(self, dout, learning_rate):
        return dout * self.mask

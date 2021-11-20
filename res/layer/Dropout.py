import numpy as np

from res.layer.Layer import Layer


class Dropout(Layer):
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        if not 0 <= p <= 1:
            raise ValueError("Not a valid probability")

    def forward(self, X, is_train=True):
        if is_train:
            self.mask = (np.random.rand(*X.shape) < self.p) / self.p
            return X * self.mask
        else:
            return X * (1 - self.p)

    def backward(self, grad):
        return grad * self.mask

    def update(self, lr):
        pass

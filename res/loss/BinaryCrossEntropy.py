import numpy as np


class BinaryCrossEntropy():
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        self.loss = np.sum(np.log(1 + np.exp(-y_true * y_pred)))
        return self.loss

    def prime(self, y_pred, y_true):
        self.grad = -y_true / (1 + np.exp(y_pred))
        return self.grad

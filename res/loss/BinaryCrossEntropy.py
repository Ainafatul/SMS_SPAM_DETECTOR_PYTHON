import numpy as np

from res.loss.Loss import Loss


class BinaryCrossEntropy(Loss):

    def __call__(self, y_pred, y_true):
        return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    def gradient(self, y_pred, y_true):
        return -y_true / (y_pred + 1e-7) + (1 - y_true) / (1 - y_pred + 1e-7)

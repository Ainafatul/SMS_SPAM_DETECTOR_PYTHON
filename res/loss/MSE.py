import numpy as np

from res.loss.Loss import Loss


class MSE(Loss):

    def __call__(self, y_pred, y_true):
        return (y_pred - y_true) ** 2

    def gradient(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / len(y_pred)

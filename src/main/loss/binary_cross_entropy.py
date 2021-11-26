import numpy as np

from loss import Loss


class BinaryCrossEntropy(Loss):

    def calc(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return np.mean(np.where(y_true == 1, -np.log(y_pred), -np.log(1 - y_pred)))

    def derivative(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return np.where(y_true == 1, -1 / y_pred, 1 / (1 - y_pred))

import numpy as np

from loss import Loss


class BinaryCrossEntropy(Loss):

    def calc(self, y_pred, y_true):
        return np.sum(np.where(y_true == 1, -np.log(y_pred), -np.log(1 - y_pred)))

    def derivative(self, y_pred, y_true):
        return np.where(y_true == 1, -1 / y_pred, 1 / (1 - y_pred))

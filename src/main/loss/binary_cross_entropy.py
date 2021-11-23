import numpy as np

from loss import Loss


class BinaryCrossEntropy(Loss):

    def calc(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        if y_true == 1:
            return -np.log(y_pred)
        else:
            return -np.log(1 - y_pred)

    def derivative(self, y_pred, y_true):
        if y_true == 1:
            return -1 / y_pred
        else:
            return 1 / (1 - y_pred)


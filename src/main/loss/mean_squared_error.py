import numpy as np

from loss import Loss


class MeanSquaredError(Loss):
    def calc(self, y_pred, y_true):
        return np.mean(np.power(y_true - y_pred, 2))

    def derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size

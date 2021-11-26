import numpy as np

from loss import Loss


class MeanSquaredError(Loss):
    def calc(self, y_pred, y_true):
        return np.sum((y_true - y_pred) ** 2)

    def derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.shape[0]

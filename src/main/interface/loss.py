import numpy as np


class Loss:

    def __call__(self, y_pred, y_true, derivative=False):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if derivative:
            return self.derivative(y_pred, y_true)
        return self.calc(y_pred, y_true)

    def calc(self, y_pred, y_true):
        raise NotImplementedError

    def derivative(self, y_pred, y_true):
        raise NotImplementedError

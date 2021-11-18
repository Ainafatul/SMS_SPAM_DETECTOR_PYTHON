import numpy as np

from res.loss.Loss import Loss


class BinaryCrossEntropy(Loss):

    def calculate(self, y_pred, y_true):
        return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)


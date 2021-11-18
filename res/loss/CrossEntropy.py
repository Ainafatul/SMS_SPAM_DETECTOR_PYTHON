import numpy as np

from res.loss.Loss import Loss


# Error For Some Unknown Reasoned
class CrossEntropy(Loss):

    def calculate(self, y_pred, y_true):
        return -y_true * np.log(y_pred)

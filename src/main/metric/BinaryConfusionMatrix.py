import numpy as np


class BinaryConfusionMatrix:

    def __init__(self, threshold=0.5):
        self.matrix = np.zeros((2, 2))
        self.threshold = threshold

    def update(self, y_pred, y_true):
        for y_p, y_t in zip(y_pred, y_true):
            self.matrix[int(y_p > self.threshold)][y_t] += 1

    def get_accuracy(self):
        return np.sum(np.diag(self.matrix)) / np.sum(self.matrix)

    def reset(self):
        self.matrix = np.zeros((2, 2))


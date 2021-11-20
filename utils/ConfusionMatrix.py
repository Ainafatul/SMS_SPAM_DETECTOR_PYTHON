import numpy as np


class ConfusionMatrix:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.matrix = np.zeros((n_classes, n_classes))

    def add(self, prediction, actual):
        self.matrix[int(prediction[0] > .5), int(actual)] += 1

    def add_batch(self, predictions, actuals):
        for prediction, actual in zip(predictions, actuals):
            self.add(prediction, actual)

    def accuracy(self):
        return np.sum(np.diag(self.matrix)) / np.sum(self.matrix)

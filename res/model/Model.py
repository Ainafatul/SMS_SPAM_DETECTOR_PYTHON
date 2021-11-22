import time

import numpy as np

from res.loss import MeanSquaredError
from utils.ConfusionMatrix import ConfusionMatrix


class Model:
    def __init__(self, loss: MeanSquaredError):
        self.layers = []
        self.loss = loss

    def __call__(self, x):
        return self.predict(x)

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def fit(self, x_train, y_train,batch_size, epochs, learning_rate):
        samples = len(x_train)
        history = []
        for i in range(epochs):
            matrix = ConfusionMatrix(2)
            loss = []

            for j in range(0, samples, batch_size):
                x_batch = x_train[j:j + batch_size]
                y_batch = y_train[j:j + batch_size]
                _, logit, loss_batch = self.train_step(x_batch, y_batch, learning_rate)

                loss.append(loss_batch)
                matrix.add_batch(logit, y_batch)
            history.append(np.mean(loss))
            print(f'epoch {i + 1}/{epochs}\n '
                  f'Loss:{np.mean(loss):.4f}, '
                  f'Acc:{matrix.accuracy():.4f}, '
                  f'TP:{matrix.matrix[0][0]}, '
                  f'FP:{matrix.matrix[0][1]}, '
                  f'TN:{matrix.matrix[1][1]}, '
                  f'FN: {matrix.matrix[1][0]}')
        return history

    def train_step(self, x_train, y_train, learning_rate):
        logit = self.predict(
            x_train)

        loss = self.loss.forward(y_train, logit)
        error = self.loss.prime(y_train, logit)

        for layer in reversed(self.layers):
            error = layer.backward(error)

        for layer in self.layers:
            layer.update(learning_rate)

        return y_train, logit, loss

from abc import ABC, abstractmethod

import numpy as np


class Model:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def update(self, grad, learning_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
        return grad

    def train(self, x, y, epochs=10, batch_size=1):
        history = []
        acc = BinaryAccuracy()
        for epoch in range(epochs):
            batch_history = []
            print(f"\nEpoch: {epoch + 1}/{epochs}")
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                logit = self(x_batch)
                grad = self.loss.gradient(logit, y_batch)
                self.update(grad, self.learning_rate)
                loss = self.loss(logit, y_batch)

                batch_history.append(loss)
                print(f"\r{i + batch_size}/{len(y)} - loss : {loss}, acc : {acc(logit, y_batch)}", end='')
            history.append(np.mean(batch_history))
        return history

    def compile(self, loss, learning_rate):
        self.loss = loss
        self.learning_rate = learning_rate


class Metric(ABC):

    @abstractmethod
    def __call__(self, y_pred, y_true):
        pass


class BinaryAccuracy(Metric):

    def __call__(self, y_pred, y_true):
        total = len(y_true)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(total):
            if y_pred[i] > 0.5 and y_true[i]:
                tp += 1
            elif y_pred[i] > 0.5 and not y_true[i]:
                fp += 1
            elif y_pred[i] < 0.5 and y_true[i]:
                fn += 1
            elif y_pred[i] < 0.5 and not y_true[i]:
                tn += 1
        # print(f"Confusion TP: {tp}, TN:{tn}, FP:{fp}, FN:{fn}")
        # print((tp + tn) / total)
        return np.mean(np.argmax(y_pred > .5, axis=1) == np.argmax(y_true, axis=1))

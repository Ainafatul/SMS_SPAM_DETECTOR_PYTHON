import time

import numpy as np

from BinaryConfusionMatrix import BinaryConfusionMatrix


class Sequential:
    confusion_matrix = BinaryConfusionMatrix()

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def batch_generator(self, x, y, batch_size):
        samples = len(x)

        index = np.arange(samples)
        np.random.shuffle(index)
        x = x[index]
        y = y[index]

        for i in range(0, samples, batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size]

    def evaluate(self, x_test, y_test):
        metric = BinaryConfusionMatrix()
        output = self.predict(x_test)
        loss = self.loss(output, y_test)
        metric.update(output, y_test)
        return loss, self.confusion_matrix.get_accuracy()

    def fit(self, training, val=None, epochs=8, batch_size=32, decay=.001):
        x_train, y_train = training

        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        for i in range(epochs):
            err = 0
            start = time.time_ns()
            for x, y in self.batch_generator(x_train, y_train, batch_size):
                logit = x
                for layer in self.layers:
                    logit = layer.forward(logit)

                err += self.loss(logit, y)

                error = self.loss(logit, y, derivative=True)
                for layer in reversed(self.layers):
                    error = layer.backward(error, self.learning_rate)

                self.confusion_matrix.update(logit, y)

            if val is not None:
                x_val, y_val = val
                val_loss, val_acc = self.evaluate(x_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)

            self.learning_rate *= (1 - decay)
            history['loss'].append(err)
            history['accuracy'].append(self.confusion_matrix.get_accuracy())
            print(f'epoch {i + 1}/{epochs} - time:{round((time.time_ns() - start) / 1e+6)}ms')
            print(f'lr :{self.learning_rate:0.5f}, loss :{err:0.4f}, acc :{self.confusion_matrix.get_accuracy():0.4f}')
            print(
                f'TP :{self.confusion_matrix.matrix[1, 1]}, TN :{self.confusion_matrix.matrix[0, 0]}, FP :{self.confusion_matrix.matrix[1, 0]}, FN :{self.confusion_matrix.matrix[0, 1]}')
            self.confusion_matrix.reset()
        return history

    def compile(self, loss, lr=0.001):
        self.loss = loss
        self.learning_rate = lr

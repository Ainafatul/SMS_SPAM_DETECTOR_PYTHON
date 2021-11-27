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
            x = layer.forward(x, training=False)
        return x

    # mengatur batch untuk training dan validasi
    def batch_generator(self, x, y, batch_size):
        # inisialisasi data yang digunakan
        samples = len(x)

        # inisialisasi pemilihan sampel dan data dibuat random
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
        for e in range(epochs):
            err = 0
            start = time.time_ns()
            for x, y in self.batch_generator(x_train, y_train, batch_size):
                logit = x

                for layer in self.layers:
                    logit = layer.forward(logit, training=True)

                err += self.loss(logit, y)

                error = self.loss(logit, y, derivative=True)
                for i, layer in enumerate(reversed(self.layers)):
                    error = layer.backward(error, self.learning_rate)

                self.confusion_matrix.update(logit, y)

            self.learning_rate *= (1 - decay)
            history['loss'].append(err)
            history['accuracy'].append(self.confusion_matrix.get_accuracy())
            print(f'epoch {e + 1}/{epochs} - time:{round((time.time_ns() - start) / 1e+6)}ms')
            print('Training')
            print(f'lr :{self.learning_rate:0.5f}, loss :{err:0.4f}, acc :{self.confusion_matrix.get_accuracy():0.4f}')
            print(f'TP :{self.confusion_matrix.matrix[1, 1]},'
                  f'TN :{self.confusion_matrix.matrix[0, 0]},'
                  f'FP :{self.confusion_matrix.matrix[1, 0]},'
                  f'FN :{self.confusion_matrix.matrix[0, 1]}')
            if val is not None:
                x_val, y_val = val
                val_loss, val_acc = self.evaluate(x_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                print('Validation')
                print(f'lr_val :{self.learning_rate:0.5f}, loss_val :{err:0.4f}, acc_val :{val_acc:0.4f}')
                print(f'TP_val :{self.confusion_matrix.matrix[1, 1]},'
                      f'TN_val :{self.confusion_matrix.matrix[0, 0]},'
                      f'FP_val :{self.confusion_matrix.matrix[1, 0]},'
                      f'FN_val :{self.confusion_matrix.matrix[0, 1]}')
            self.confusion_matrix.reset()
        return history

    def compile(self, loss, lr=0.01):
        self.loss = loss
        self.learning_rate = lr

import numpy as np

from binary_cross_entropy import BinaryCrossEntropy
from mean_squared_error import MeanSquaredError


class BinaryConfusionMatrix:

    def __init__(self, threshold=0.5):
        self.matrix = np.zeros((2, 2))
        self.threshold = threshold

    def update(self, y_pred, y_true):
        for y_p, y_t in zip(y_pred, y_true):
            self.matrix[int(y_p > self.threshold)][y_t] += 1

    def get_accuracy(self):
        return np.sum(np.diag(self.matrix)) / np.sum(self.matrix)

    def print_matrix(self):
        print(f'TP : {self.matrix[1, 1]}, TN : {self.matrix[0, 0]}, FP : {self.matrix[1, 0]}, FN : {self.matrix[0, 1]}')


class Sequential:
    loss = BinaryCrossEntropy()
    confusion_matrix = BinaryConfusionMatrix()

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return np.array(result)

    def batch_generator(self, x, y, batch_size):
        samples = len(x)

        index = np.arange(samples)
        np.random.shuffle(index)
        x = x[index]
        y = y[index]

        for i in range(0, samples, batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size]

    def fit(self, x_train, y_train, epochs=8, batch_size=32, learning_rate=.001, decay=.001):
        samples = len(x_train)
        history = {'loss': [], 'accuracy': []}
        for i in range(epochs):
            err = 0
            for x, y in self.batch_generator(x_train, y_train, batch_size):
                output = x
                for layer in self.layers:
                    output = layer.forward(output)

                err += self.loss(output, y)

                error = self.loss(output, y, True)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

                self.confusion_matrix.update(output, y)
            err /= samples
            learning_rate *= (1 - decay)
            history['loss'].append(np.mean(err))
            history['accuracy'].append(self.confusion_matrix.get_accuracy())
            print(f'epoch {i + 1}')
            print(f'lr :{learning_rate:0.5f}, {epochs}, loss :{np.mean(err):0.4f}, acc :{self.confusion_matrix.get_accuracy():0.4f}')
        return history

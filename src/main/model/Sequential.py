import numpy as np

from binary_cross_entropy import BinaryCrossEntropy


class Sequential:
    loss = BinaryCrossEntropy()

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

    def fit(self, x_train, y_train, epochs, learning_rate, decay=.001):
        samples = len(x_train)

        loss = []
        for i in range(epochs):
            err = 0
            for x, y in zip(x_train, y_train):
                x, y = np.array([x]), np.array([y])
                output = x
                for layer in self.layers:
                    output = layer.forward(output)

                err += self.loss(output, y)

                error = self.loss(output, y, True)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
            err /= samples
            loss.append(np.mean(err))
            learning_rate *= (1 - decay)
            print(f'epoch {i + 1}/{epochs}, error :{np.mean(err):0.4f}, lr :{learning_rate:0.5f}')
        return loss

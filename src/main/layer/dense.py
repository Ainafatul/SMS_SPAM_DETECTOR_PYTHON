from layer import Layer
import numpy as np


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.zeros(output_size) - 0.5

    def forward(self, x):
        self.input = x
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        batch_size = self.input.shape[0]
        input_error = np.dot(output_error, self.weights.T)

        weights_error = (1 / batch_size) * np.dot(self.input.T, output_error) + (.7/batch_size) * self.weights

        self.weights -= weights_error * learning_rate
        self.bias -= np.sum(output_error, axis=0) * learning_rate
        return input_error



from res.layers.Layer import Layer
import numpy as np


class Dense(Layer):

    def __init__(self, input: int, output: int):
        self.weights = np.random.normal(loc=.0, size=(input, output))
        self.bias = np.zeros(shape=output)

    def forward(self, x):
        self.logit = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, error):
        input_error = np.dot(error, self.weights.T)
        weight_error = np.dot(self.logit.T, error)
        self.weights -= weight_error * .0001
        self.bias -= np.mean(error, axis=0) * self.logit[0].shape * .0001
        return input_error


class ReLu(Layer):
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, x):
        return np.where(x > 0, 1, 0)


class Sigmoid(Layer):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))

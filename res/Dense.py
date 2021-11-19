from res.layers.Layer import Layer
import numpy as np


class Dense(Layer):

    def __init__(self, input: int, output: int):
        self.weights = np.random.uniform(size=(input, output))
        self.bias = np.zeros(shape=output)

    def forward(self, x):
        self.logit = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad, learning_rate):
        input_error = np.dot(grad, self.weights.T)
        weight_error = np.dot(self.logit.T, grad)
        self.weights -= weight_error * learning_rate
        self.bias -= np.mean(grad, axis=0) * learning_rate
        return input_error


class ReLu(Layer):
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, x, learning_rate):
        return np.where(x > 0, 1, 0)


class Sigmoid(Layer):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x, learning_rate):
        return self.forward(x) * (1 - self.forward(x))


class Linear(Layer):
    def forward(self, x):
        return x

    def backward(self, x, learning_rate):
        return np.ones(x.shape)

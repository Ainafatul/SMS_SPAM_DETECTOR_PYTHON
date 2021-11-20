import numpy as np

from res.layer.Layer import Layer


class Dense(Layer):

    def __init__(self, input_shape, units):
        self.input_shape = input_shape
        self.units = units
        self.weights = np.random.randn(input_shape, units)
        self.bias = np.random.randn(1, units)

    # function to calculate the output of the layer
    # check if the input is 2D numpy array if not expand it
    def forward(self, input_data):
        self.input = input_data
        output = np.dot(input_data, self.weights) + self.bias
        return output

    def backward(self, error):
        self.weights_error = np.dot(self.input.T, error)
        self.bias_error = np.mean(error, axis=0)
        return np.dot(error, self.weights.T)

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_error
        self.bias -= learning_rate * self.bias_error

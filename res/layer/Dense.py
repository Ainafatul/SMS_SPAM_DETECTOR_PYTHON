import numpy as np

from res.layer.Layer import Layer


class Dense(Layer):

    def __init__(self, units, input_shape):
        self.input_shape = input_shape
        self.units = units
        self.weights = np.random.uniform(size=(input_shape, units))
        self.bias = np.zeros(units)

    def forward(self, input_data):
        self.input = input_data
        return self.input @ self.weights + self.bias

    def backward(self, error):
        error = np.mean(error, axis=1)
        print("Dense backward")
        print("error: ", error.shape)
        print("weights: ", self.weights.shape)
        print("input: ", self.input.shape)
        print(f"bias: ",self.bias.shape)

        self.weights_error = self.input.T @ error
        self.bias_error = error

        print("weights_error: ", self.weights_error.shape)
        print("bias_error: ", self.bias_error.shape)
        return error @ self.weights.T

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_error
        self.bias -= learning_rate * self.bias_error

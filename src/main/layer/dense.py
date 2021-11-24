from layer import Layer
import numpy as np


class Dense(Layer):
    input = None
    input_shape = None
    output_shape = None

    def __init__(self, units, input_shape=None):
        if input_shape is not None:
            self.input_shape = input_shape[-1]
        self.output_shape = (units,)

        self.weights = None
        self.bias = None

    def forward(self, x):
        self.input = x
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)

        weights_error = np.dot(self.input.T, output_error)

        self.weights -= gradient_clip(weights_error, 1) * learning_rate
        self.bias -= np.sum(output_error) * learning_rate
        return input_error

    def compile(self, input_shape=None):
        if input_shape is not None:
            self.input_shape = input_shape[-1]

        self.weights = np.random.randn(self.input_shape, self.output_shape[0])
        self.bias = np.random.randn(self.output_shape[0])

        print(f"Dense : {self.input_shape}->{self.output_shape}")


def gradient_clip(gradient, clip_norm):
    return np.clip(gradient, -clip_norm, clip_norm)

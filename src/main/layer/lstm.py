import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Sequential import Sequential
from binary_cross_entropy import BinaryCrossEntropy
from dense import Dense
from layer import Layer
from sigmoid import Sigmoid
from tanh import Tanh
from text_vetorization import TextVectorization


class LSTMLayer(Layer):
    input = None
    input_shape = None
    output_shape = None

    def sigmoid(self, x):
        x = x.clip(-700, 700)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh_derivative(self, x):
        return 1 - (np.tanh(x)) ** 2

    def tanh(self, x):
        return np.tanh(x)

    def __init__(self, units, input_shape=None, return_sequence=False):
        if input_shape is not None:
            self.input_shape = input_shape
        self.units = units
        self.return_sequence = return_sequence

    def _forward(self, x):
        self.batch_size, self.time_step, self.features = x.shape

        self.output = np.zeros((self.batch_size, self.time_step, self.units))
        self.state = np.zeros((self.batch_size, self.time_step, self.units))

        self.f = np.zeros((self.batch_size, self.time_step, self.units))
        self.i = np.zeros((self.batch_size, self.time_step, self.units))
        self.o = np.zeros((self.batch_size, self.time_step, self.units))
        self.a = np.zeros((self.batch_size, self.time_step, self.units))

        self.X = np.zeros((self.batch_size, self.time_step, self.features + self.units))
        self.x = x

        prev_state = np.zeros((self.batch_size, self.units))
        prev_output = np.zeros((self.batch_size, self.units))

        for t in range(self.time_step):
            x_t = np.column_stack((prev_output, x[:, t, :]))
            self.X[:, t, :] = x_t

            self.a[:, t] = self.tanh(np.dot(x_t, self.Wa) + self.Ba)
            self.i[:, t] = self.sigmoid(np.dot(x_t, self.Wi) + self.Bi)
            self.f[:, t] = self.sigmoid(np.dot(x_t, self.Wf) + self.Bf)
            self.o[:, t] = self.sigmoid(np.dot(x_t, self.Wo) + self.Bo)

            self.state[:, t, :] = (self.f[:, t] * prev_state) + (self.i[:, t, :] * self.a[:, t])
            self.output[:, t, :] = self.o[:, t] * self.tanh(self.state[:, t])

            prev_state = self.state[:, t, :]
            prev_output = self.output[:, t, :]

        if self.return_sequence:
            return self.output
        return self.output.mean(axis=1)

    def _backward(self, d_loss, learning_rate):
        delta_output_next = np.zeros((self.batch_size, self.units))
        delta_state_next = np.zeros((self.batch_size, self.units))

        dWa = np.zeros_like(self.Wa)
        dWi = np.zeros_like(self.Wi)
        dWf = np.zeros_like(self.Wf)
        dWo = np.zeros_like(self.Wo)

        dBa = np.zeros_like(self.Ba)
        dBi = np.zeros_like(self.Bi)
        dBf = np.zeros_like(self.Bf)
        dBo = np.zeros_like(self.Bo)

        for t in reversed(range(self.time_step)):
            prev_state = self.state[:, t, :]
            if t == 0:
                prev_state = np.zeros_like(prev_state)

            delta_out = d_loss + delta_output_next

            delta_state = delta_out * self.o[:, t, :] * self.tanh_derivative(self.state[:, t, :]) + delta_state_next
            delta_a = delta_state * self.i[:, t, :] * self.tanh_derivative(self.a[:, t, :])
            delta_i = delta_state * self.a[:, t, :] * self.sigmoid_derivative(self.i[:, t, :])
            delta_f = delta_state * prev_state * self.sigmoid_derivative(self.f[:, t, :])
            delta_o = delta_out * self.tanh(self.state[:, t, :]) * self.sigmoid_derivative(self.o[:, t, :])

            dWa -= self.X[:, t].T.dot(delta_a)
            dWi -= self.X[:, t].T.dot(delta_i)
            dWf -= self.X[:, t].T.dot(delta_f)
            dWo -= self.X[:, t].T.dot(delta_o)

            dBa -= delta_a.mean(axis=0, keepdims=True)
            dBi -= delta_i.mean(axis=0, keepdims=True)
            dBf -= delta_f.mean(axis=0, keepdims=True)
            dBo -= delta_o.mean(axis=0, keepdims=True)

            dX = np.dot(delta_a , self.Wa.T) + np.dot(delta_i , self.Wi.T) + np.dot(delta_f , self.Wf.T) + np.dot(delta_o , self.Wo.T)

            delta_output_next = dX[:, :self.units]
            delta_state_next = delta_state * self.f[:, t, :]

        clip = lambda x: np.nan_to_num(np.sum(x / self.batch_size, axis=0, keepdims=True))

        self.Wa += learning_rate * clip(dWa)
        self.Wi += learning_rate * clip(dWi)
        self.Wf += learning_rate * clip(dWf)
        self.Wo += learning_rate * clip(dWo)

        self.Ba += learning_rate * dBa
        self.Bi += learning_rate * dBi
        self.Bf += learning_rate * dBf
        self.Bo += learning_rate * dBo

        return delta_output_next

    def compile(self, input_shape=None):
        if input_shape is not None:
            self.input_shape = input_shape

        self.time_step, self.features = self.input_shape

        if self.return_sequence:
            self.output_shape = (self.time_step, self.units,)
        else:
            self.output_shape = (self.units,)

        self.Wf = np.random.randn(self.features + self.units, self.units)
        self.Wi = np.random.randn(self.features + self.units, self.units)
        self.Wa = np.random.randn(self.features + self.units, self.units)
        self.Wo = np.random.randn(self.features + self.units, self.units)

        self.Bf = np.zeros(shape=(1, self.units))
        self.Bi = np.zeros(shape=(1, self.units))
        self.Ba = np.zeros(shape=(1, self.units))
        self.Bo = np.zeros(shape=(1, self.units))

        print(f"LSTM : {self.input_shape} -> {self.output_shape}")


if __name__ == '__main__':
    time_step = 3
    feature = 1

    layers = [
        LSTMLayer(32, input_shape=(time_step, feature), return_sequence=False),
        Dense(8),
        Sigmoid()
    ]

    for i, layer in enumerate(layers):
        if i != 0:
            layer.compile(input_shape=layers[i - 1].output_shape)
            continue
        layer.compile()

    loss = BinaryCrossEntropy()

    errors = []

    x = np.array([[[1], [2], [1]], [[1], [2], [4]], [[3], [4], [1]], [[4], [4], [1]]])
    y = np.array([[1], [0], [0], [1]])

    for i in range(10240):
        p = x
        for layer in layers:
            p = layer.forward(p)

        error = loss(p, y, derivative=True)

        for layer in reversed(layers):
            error = layer.backward(error, 0.01)

        err = loss(p, y)

        print(f'Epoch {i}, error={err:0.4}')
        errors.append(err)

    for layer in layers:
        x = layer.forward(x)

    plt.plot(errors, label='error')
    plt.legend()
    plt.show()

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
    tanh = Tanh()
    sigmoid = Sigmoid()

    def __init__(self, units, input_shape=None, return_sequence=False):
        if input_shape is not None:
            self.input_shape = input_shape
        self.units = units
        self.return_sequence = return_sequence

    def forward(self, x):
        batch_size = x.shape[0]

        self.out = np.zeros((batch_size, self.time_step, self.units))
        self.state = np.zeros((batch_size, self.time_step, self.units))

        self.f = np.zeros((self.time_step, batch_size, self.units))
        self.i = np.zeros((self.time_step, batch_size, self.units))
        self.o = np.zeros((self.time_step, batch_size, self.units))
        self.a = np.zeros((self.time_step, batch_size, self.units))

        self.x = x

        out = np.zeros((batch_size, self.units))
        prev_state = np.zeros((batch_size, self.units))

        for t in range(self.time_step):
            x_t = x[:, t, :]
            self.a[t] = self.tanh(np.dot(x_t, self.Wa) + np.dot(out, self.Ua))
            self.i[t] = self.sigmoid(np.dot(x_t, self.Wi) + np.dot(out, self.Ui))
            self.f[t] = self.sigmoid(np.dot(x_t, self.Wf) + np.dot(out, self.Uf))
            self.o[t] = self.sigmoid(np.dot(x_t, self.Wo) + np.dot(out, self.Uo))

            self.state[:, t, :] = self.a[t] * self.i[t] * self.f[t] * prev_state
            self.out[:, t, :] = self.tanh(self.state[:, t, :]) * self.o[t]

            prev_state = self.state[:, t, :]
            out = self.out[:, t, :]

        if self.return_sequence:
            return self.out
        return out

    def backward(self, d_loss, learning_rate):
        delta_out_next = np.zeros_like(self.out[:, 0])
        d_out = np.zeros_like(self.out[:, 0])
        f_next = np.zeros_like(self.f[0])
        for t in reversed(range(self.time_step)):
            delta_out = d_loss + d_out

            delta_state = delta_out * self.o[t] * (1 - self.out[:, t, :] ** 2) + delta_out_next * f_next
            delta_a = delta_state * self.i[t] * (1 - self.a[t] ** 2)
            delta_i = delta_state * self.a[t] * self.i[t] * (1 - self.i[t])
            delta_f = delta_state * self.state[:, t - 1, :] * self.f[t] * (1 - self.f[t])
            delta_o = delta_out * self.tanh(self.state[:, t, :]) * self.o[t] * (1 - self.o[t])

            f = lambda x: np.clip(x, -1, 1)

            self.Wa -= learning_rate * f(np.dot(self.x[:, t, :].T, delta_a))
            self.Ua -= learning_rate * f(np.dot(delta_out_next.T, delta_a))
            self.Wf -= learning_rate * f(np.dot(self.x[:, t, :].T, delta_f))
            self.Uf -= learning_rate * f(np.dot(delta_out_next.T, delta_f))
            self.Wi -= learning_rate * f(np.dot(self.x[:, t, :].T, delta_i))
            self.Ui -= learning_rate * f(np.dot(delta_out_next.T, delta_i))
            self.Wo -= learning_rate * f(np.dot(self.x[:, t, :].T, delta_o))
            self.Uo -= learning_rate * f(np.dot(delta_out_next.T, delta_o))

            f_next = self.f[t]
            delta_out_next = delta_out
            d_out = np.dot(delta_a, self.Ua.T) + np.dot(delta_i, self.Ui.T) + np.dot(delta_f, self.Uf.T) + np.dot(delta_o, self.Uo.T)

        delta_input = np.dot(delta_a, self.Wa.T) + np.dot(delta_f, self.Wf.T) + np.dot(delta_i, self.Wi.T) + np.dot(delta_o, self.Wo.T)
        return delta_input

    def compile(self, input_shape=None):
        if input_shape is not None:
            self.input_shape = input_shape

        self.time_step, self.features = self.input_shape

        if self.return_sequence:
            self.output_shape = (self.time_step, self.units,)
        else:
            self.output_shape = (self.units,)

        self.Wf = np.random.randn(self.features, self.units) * .001
        self.Wi = np.random.randn(self.features, self.units) * .001
        self.Wa = np.random.randn(self.features, self.units) * .001
        self.Wo = np.random.randn(self.features, self.units) * .001

        self.Uf = np.random.randn(self.units, self.units) * .001
        self.Ui = np.random.randn(self.units, self.units) * .001
        self.Ua = np.random.randn(self.units, self.units) * .001
        self.Uo = np.random.randn(self.units, self.units) * .001


if __name__ == '__main__':
    time_step = 32
    feature = 4

    x = np.ones((2, time_step, feature))

    layer_1 = LSTMLayer(16, input_shape=(time_step, feature), return_sequence=True)
    layer_2 = LSTMLayer(8, return_sequence=False)
    layer_3 = Dense(1)

    layer_1.compile()
    layer_2.compile(layer_1.output_shape)
    layer_3.compile(layer_2.output_shape)

    losses = BinaryCrossEntropy()

    for i in range(32):
        x_1 = layer_1.forward(x)
        x_2 = layer_2.forward(x_1)
        x_3 = layer_3.forward(x_2)

        d_loss = losses(x_3, derivative=True)
        print(f'd_loss shape: {d_loss.shape}')

        d3 = layer_3.backward(d_loss, 0.1)
        print(f'd3 shape: {d3.shape}')
        d2 = layer_2.backward(d3, 0.1)
        print(f'd2 shape: {d2.shape}')
        d1 = layer_1.backward(d2, 0.1)
        print(f'd1 shape: {d1.shape}')

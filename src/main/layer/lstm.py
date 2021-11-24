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
        delta_state_next = np.zeros_like(self.state[:, 0])

        dWa = np.zeros((self.time_step, *self.Wa.shape))
        dWi = np.zeros((self.time_step, *self.Wi.shape))
        dWf = np.zeros((self.time_step, *self.Wf.shape))
        dWo = np.zeros((self.time_step, *self.Wo.shape))

        dUa = np.zeros((self.time_step, *self.Ua.shape))
        dUi = np.zeros((self.time_step, *self.Ui.shape))
        dUf = np.zeros((self.time_step, *self.Uf.shape))
        dUo = np.zeros((self.time_step, *self.Uo.shape))

        for t in reversed(range(self.time_step)):
            delta_out = d_loss + delta_out_next

            delta_state = delta_out * self.o[t] * self.tanh(self.state[:, t, :], derivative=True) + delta_state_next
            delta_a = delta_state * self.i[t] * self.tanh(self.a[t], derivative=True)
            delta_i = delta_state * self.a[t] * self.sigmoid(self.i[t], derivative=True)
            delta_f = delta_state * self.state[:, t - 1, :] * self.sigmoid(self.f[t], derivative=True)
            delta_o = delta_out * self.tanh(self.state[:, t, :]) * self.sigmoid(self.o[t], derivative=True)

            dWa[t] = np.dot(self.x[:, t, :].T, delta_a)
            dWi[t] = np.dot(self.x[:, t, :].T, delta_i)
            dWf[t] = np.dot(self.x[:, t, :].T, delta_f)
            dWo[t] = np.dot(self.x[:, t, :].T, delta_o)

            dUa[t] = np.dot(delta_out_next.T, delta_a)
            dUi[t] = np.dot(delta_out_next.T, delta_i)
            dUf[t] = np.dot(delta_out_next.T, delta_f)
            dUo[t] = np.dot(delta_out_next.T, delta_o)

            delta_out_next = np.dot(delta_a, self.Ua.T) + np.dot(delta_i, self.Ui.T) + np.dot(delta_f, self.Uf.T) + np.dot(delta_o, self.Uo.T)
            delta_state_next = delta_state * self.f[t]

        self.Wa -= learning_rate * np.sum(dWa, axis=0)
        self.Wi -= learning_rate * np.sum(dWi, axis=0)
        self.Wf -= learning_rate * np.sum(dWf, axis=0)
        self.Wo -= learning_rate * np.sum(dWo, axis=0)

        return delta_out_next

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

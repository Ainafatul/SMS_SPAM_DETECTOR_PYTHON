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
        self.batch_size = x.shape[0]

        self.output = np.zeros((self.batch_size, self.time_step, self.units))
        self.state = np.zeros((self.batch_size, self.time_step, self.units))

        self.f = np.zeros((self.time_step, self.batch_size, self.units))
        self.i = np.zeros((self.time_step, self.batch_size, self.units))
        self.o = np.zeros((self.time_step, self.batch_size, self.units))
        self.a = np.zeros((self.time_step, self.batch_size, self.units))

        self.X = np.zeros((self.time_step, self.batch_size, self.features + self.units))
        self.x = x

        prev_state = np.zeros((self.batch_size, self.units))
        prev_output = np.zeros((self.batch_size, self.units))

        for t in range(self.time_step):
            x_t = np.column_stack((prev_output, x[:, t, :]))
            self.X[t] = x_t

            self.a[t] = self.tanh(x_t @ self.Wa + self.Ba)
            self.i[t] = self.sigmoid(x_t @ self.Wi + self.Bi)
            self.f[t] = self.sigmoid(x_t @ self.Wf + self.Bf)
            self.o[t] = self.sigmoid(x_t @ self.Wo + self.Bo)

            self.state[:, t, :] = self.f[t] * prev_state + self.i[t] * self.a[t]
            self.output[:, t, :] = self.o[t] * self.tanh(self.state[:, t, :])

            prev_state = self.state[:, t, :]
            prev_output = self.output[:, t, :]

        if self.return_sequence:
            return self.output
        return self.output.mean(axis=1)

    def backward(self, d_loss, learning_rate):
        print(f"LSTM : {self.input_shape} -> {self.output_shape}")
        print(f"LSTM : {d_loss.shape}")
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

            delta_state = delta_out * self.o[t] * self.tanh(self.state[:, t, :], derivative=True) + delta_state_next
            delta_a = delta_state * self.i[t] * self.tanh(self.a[t], derivative=True)
            delta_i = delta_state * self.a[t] * self.sigmoid(self.i[t], derivative=True)
            delta_f = delta_state * prev_state * self.sigmoid(self.f[t], derivative=True)
            delta_o = delta_out * self.tanh(self.state[:, t, :]) * self.sigmoid(self.o[t], derivative=True)

            dWa += self.X[t].T @ delta_a
            dWi += self.X[t].T @ delta_i
            dWf += self.X[t].T @ delta_f
            dWo += self.X[t].T @ delta_o

            dBa += delta_a.mean(axis=0)
            dBi += delta_i.mean(axis=0)
            dBf += delta_f.mean(axis=0)
            dBo += delta_o.mean(axis=0)

            dX = delta_a @ self.Wa.T + delta_i @ self.Wi.T + delta_f @ self.Wf.T + delta_o @ self.Wo.T

            delta_output_next = dX[:, :self.units]
            delta_state_next = delta_state * self.f[t]

        clip = lambda x: x

        self.Wa -= learning_rate * clip(dWa)
        self.Wi -= learning_rate * clip(dWi)
        self.Wf -= learning_rate * clip(dWf)
        self.Wo -= learning_rate * clip(dWo)

        self.Ba -= learning_rate * clip(dBa)
        self.Bi -= learning_rate * clip(dBi)
        self.Bf -= learning_rate * clip(dBf)
        self.Bo -= learning_rate * clip(dBo)

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

        d_loss = losses(x_3, np.ones((2, 1)), derivative=True)
        print(f'd_loss shape: {d_loss.shape}')

        d3 = layer_3.backward(d_loss, 0.1)
        print(f'd3 shape: {d3.shape}')
        d2 = layer_2.backward(d3, 0.1)
        print(f'd2 shape: {d2.shape}')
        d1 = layer_1.backward(d2, 0.1)
        print(f'd1 shape: {d1.shape}')

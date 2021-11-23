import numpy as np
import pandas as pd

from Sequential import Sequential
from layer import Layer
from sigmoid import Sigmoid
from tanh import Tanh
from text_vetorization import TextVectorization


class LSTMLayer(Layer):
    tanh = Tanh()
    sigmoid = Sigmoid()

    def __init__(self, units, input_shape=None, return_sequence=False):
        self.units = units
        self.return_sequence = return_sequence
        self.batch, self.time_step, self.features = input_shape
        self.Wxf = np.random.randn(self.features, units) * .001
        self.Wxi = np.random.randn(self.features, units) * .001
        self.Wxc = np.random.randn(self.features, units) * .001
        self.Wxo = np.random.randn(self.features, units) * .001

        self.Whf = np.random.randn(units, units) * .001
        self.Whi = np.random.randn(units, units) * .001
        self.Whc = np.random.randn(units, units) * .001
        self.Who = np.random.randn(units, units) * .001

        self.states = np.zeros((self.batch, self.time_step, self.units))
        self.cells = np.zeros((self.batch, self.time_step, self.units))

        self.forget_gates = np.zeros((self.time_step, self.batch, self.units))
        self.input_gates = np.zeros((self.time_step, self.batch, self.units))
        self.output_gates = np.zeros((self.time_step, self.batch, self.units))
        self.cell_bar = np.zeros((self.time_step, self.batch, self.units))

    def forward(self, x):
        self.x = x

        h = np.zeros((self.batch, self.units))
        c = np.zeros((self.batch, self.units))

        for t in range(self.time_step):
            x_t = x[:, t, :]
            self.forget_gates[t] = self.sigmoid.forward(np.dot(self.Wxf) + np.dot(h, self.Whf))
            self.input_gates[t] = self.sigmoid.forward(np.dot(x_t, self.Wxi) + np.dot(h, self.Whi))
            self.output_gates[t] = self.sigmoid.forward(np.dot(x_t, self.Wxo) + np.dot(h, self.Who))

            self.cell_bar[t] = self.tanh.forward(np.dot(x_t, self.Wxc) + np.dot(h, self.Whc))

            self.cells[:, t, :] = self.forget_gates[t] * c + self.input_gates[t] * self.cell_bar[t]
            self.states[:, t, :] = self.output_gates[t] * self.tanh.forward(c)

            c = self.cells[:, t, :]
            h = self.states[:, t, :]

        if self.return_sequence:
            return self.states
        return h

    def backward(self, d_loss, learning_rate):
        d_output = np.zeros_like(self.states[0])
        next_state = np.zeros_like(self.states[0])
        next_forget_gate = np.zeros_like(self.forget_gates[0])
        for t in reversed(range(self.time_step)):
            delta_output = d_loss + d_output

            delta_state = delta_output * self.output_gates[t] * self.tanh.backward_activation(self.states[:, t, :]) + next_state + next_forget_gate
            delta_cell = delta_state * self.input_gates[t] * (1 - self.cell_bar[t] ** 2)
            delta_it = delta_state * self.cell_bar[t] * self.input_gates[t] * (1 - self.input_gates[t])
            delta_ft = delta_state * self.cells[:, t - 1, :] * self.forget_gates[t] * (1 - self.forget_gates[t])
            delta_ot = delta_output * self.tanh.forward(self.states[:, t, :]) * self.output_gates[t] * (1 - self.output_gates[t])

            next_forget_gate = self.forget_gates[t]
            next_state = self.states[:, t, :]

            delta_ft = np.clip(delta_ft, -1, 1)
            delta_it = np.clip(delta_it, -1, 1)
            delta_ot = np.clip(delta_ot, -1, 1)
            delta_cell = np.clip(delta_cell, -1, 1)

            self.Wxf -= learning_rate * delta_ft.mean(axis=0)
            self.Wxi -= learning_rate * delta_it.mean(axis=0)
            self.Wxo -= learning_rate * delta_ot.mean(axis=0)
            self.Wxc -= learning_rate * delta_cell.mean(axis=0)

            self.Whf -= learning_rate * delta_ft.mean(axis=0)
            self.Whi -= learning_rate * delta_it.mean(axis=0)
            self.Who -= learning_rate * delta_ot.mean(axis=0)
            self.Whc -= learning_rate * delta_cell.mean(axis=0)


if __name__ == '__main__':
    dataset = pd.read_csv('../../res/imdb.csv')
    x = dataset.iloc[:, 0].values
    y = dataset.iloc[:, -1].values

    encoder = TextVectorization(1024, 32)
    encoder.fit(x)

    x = encoder(x)
    y = np.array([1 if label == 'positive' else 0 for label in y])

    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    x = x.reshape(x.shape[0], x.shape[1], 1)
    y = y.reshape(y.shape[0], 1)

    net = Sequential()
    net.add(LSTMLayer(units=8, input_shape=(1, 32, 1), return_sequence=False))

    print(net.predict(np.array([x[:1]])).shape)

    net.fit(x, y, epochs=8, learning_rate=0.01)

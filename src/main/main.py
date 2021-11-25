import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Sequential import Sequential
from binary_cross_entropy import BinaryCrossEntropy
from dense import Dense, gradient_clip
from mean_squared_error import MeanSquaredError
from sigmoid import Sigmoid
from tanh import Tanh
from text_vetorization import TextVectorization


class LSTM:

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def __init__(self, unit, input_shape, return_sequence=False):
        self.return_sequence = return_sequence
        self.time, self.features = input_shape
        self.unit = unit

        self.W_i = np.random.randn(self.features, self.unit) * np.sqrt(1 / self.features + self.unit)
        self.W_f = np.random.randn(self.features, self.unit) * np.sqrt(1 / self.features + self.unit)
        self.W_a = np.random.randn(self.features, self.unit) * np.sqrt(1 / self.features + self.unit)
        self.W_o = np.random.randn(self.features, self.unit) * np.sqrt(1 / self.features + self.unit)

        self.U_i = np.random.randn(self.unit, self.unit) * np.sqrt(1 / self.features + self.unit)
        self.U_f = np.random.randn(self.unit, self.unit) * np.sqrt(1 / self.features + self.unit)
        self.U_a = np.random.randn(self.unit, self.unit) * np.sqrt(1 / self.features + self.unit)
        self.U_o = np.random.randn(self.unit, self.unit) * np.sqrt(1 / self.features + self.unit)

        self.B_a = np.zeros((1, self.unit))
        self.B_i = np.zeros((1, self.unit))
        self.B_f = np.zeros((1, self.unit))
        self.B_o = np.zeros((1, self.unit))

    def __gate(self, x, w, b):
        return np.dot(x, w) + b

    def forward(self, x):
        self.batch_size, time, features = x.shape
        pri
        assert features == self.features, "Input shape does not match"
        assert time == self.time, "Input shape does not match"

        self.x = np.zeros((self.time, self.batch_size, self.features))

        self.a = np.zeros((self.time, self.batch_size, self.unit))
        self.i = np.zeros((self.time, self.batch_size, self.unit))
        self.f = np.zeros((self.time, self.batch_size, self.unit))
        self.o = np.zeros((self.time, self.batch_size, self.unit))

        self.state = np.zeros((self.batch_size, self.time, self.unit))
        self.output = np.zeros((self.batch_size, self.time, self.unit))

        for t in range(self.time):
            self.x[t] = x[:, t]

            self.a[t] = np.tanh(np.dot(x[:, t], self.W_a) + np.dot(self.output[:, t - 1], self.U_a) + self.B_a)
            self.i[t] = self.sigmoid(np.dot(x[:, t], self.W_i) + np.dot(self.output[:, t - 1], self.U_i) + self.B_i)
            self.f[t] = self.sigmoid(np.dot(x[:, t], self.W_f) + np.dot(self.output[:, t - 1], self.U_f) + self.B_f)
            self.o[t] = self.sigmoid(np.dot(x[:, t], self.W_o) + np.dot(self.output[:, t - 1], self.U_o) + self.B_o)

            self.state[:, t] = (self.a[t] * self.i[t]) + (self.f[t] * self.state[:, t - 1])
            self.output[:, t] = np.tanh(self.state[:, t]) * (self.o[t])

        # print('\n\n\n')
        if self.return_sequence:
            return self.output
        return self.output[:, -1]

    def backward(self, loss, learning_rate):
        dWa = np.zeros_like(self.W_a)
        dWi = np.zeros_like(self.W_i)
        dWf = np.zeros_like(self.W_f)
        dWo = np.zeros_like(self.W_o)

        dUa = np.zeros_like(self.U_a)
        dUi = np.zeros_like(self.U_i)
        dUf = np.zeros_like(self.U_f)
        dUo = np.zeros_like(self.U_o)

        dBa = np.zeros_like(self.B_a)
        dBi = np.zeros_like(self.B_i)
        dBf = np.zeros_like(self.B_f)
        dBo = np.zeros_like(self.B_o)

        d_output_next = np.zeros_like(self.state[:, 0])
        error = loss
        for t in reversed(range(self.time)):
            if loss.ndim == 3:
                error = loss[:, t, :]
                pass
            if t == self.time - 1:
                state_next = np.zeros_like(self.state[:, 0])
                f_next = np.zeros_like(self.f[0])
            else:
                state_next = self.state[:, t + 1]
                f_next = self.f[t + 1]
            if t == 0:
                prev_state = np.zeros_like(self.state[:, 0])
            else:
                prev_state = self.state[:, t - 1]

            if t < self.time - 1:
                dUa += np.mean(d_a * self.output[:, t], axis=0, keepdims=True)
                dUi += np.mean(d_i * self.output[:, t], axis=0, keepdims=True)
                dUf += np.mean(d_f * self.output[:, t], axis=0, keepdims=True)
                dUo += np.mean(d_o * self.output[:, t], axis=0, keepdims=True)

            d_out = error + d_output_next
            d_state = d_out * self.o[t] * (1 - np.tanh(self.state[:, t]) ** 2) + state_next * f_next
            d_a = d_state * (self.i[t]) * (1 - (self.a[t] ** 2))
            d_i = d_state * (self.a[t]) * (self.i[t]) * (1 - self.i[t])
            d_f = d_state * prev_state * self.f[t] * (1 - self.f[t])
            d_o = d_out * (np.tanh(self.state[:, t])) * (self.o[t]) * (1 - self.o[t])

            print('d_a', d_a.shape)
            print('x_t', x[t].shape)

            if not self.return_sequence:
                dWa += (d_a * self.x[t]).mean(axis=0, keepdims=True).T
                dWi += (d_i * self.x[t]).mean(axis=0, keepdims=True).T
                dWf += (d_f * self.x[t]).mean(axis=0, keepdims=True).T
                dWo += (d_o * self.x[t]).mean(axis=0, keepdims=True).T
            else:
                dWa += np.dot(d_a, self.W_a.T).mean(axis=0, keepdims=True).T
                dWi += np.dot(d_i, self.W_i.T).mean(axis=0, keepdims=True).T
                dWf += np.dot(d_f, self.W_f.T).mean(axis=0, keepdims=True).T
                dWo += np.dot(d_o, self.W_o.T).mean(axis=0, keepdims=True).T

            dBa += d_a.mean(axis=0)
            dBi += d_i.mean(axis=0)
            dBf += d_f.mean(axis=0)
            dBo += d_o.mean(axis=0)

            d_output_next = (d_a.dot(self.U_a.T)) + (d_i.dot(self.U_i.T)) + (d_f.dot(self.U_f.T)) + (d_o.dot(self.U_o.T))

        self.W_a -= learning_rate * dWa
        self.W_i -= learning_rate * dWi
        self.W_f -= learning_rate * dWf
        self.W_o -= learning_rate * dWo

        self.U_a -= learning_rate * dUa
        self.U_i -= learning_rate * dUi
        self.U_f -= learning_rate * dUf
        self.U_o -= learning_rate * dUo

        # self.W_a -= learning_rate * gradient_clip(dWa, 1)
        # self.W_i -= learning_rate * gradient_clip(dWi, 1)
        # self.W_f -= learning_rate * gradient_clip(dWf, 1)
        # self.W_o -= learning_rate * gradient_clip(dWo, 1)
        #
        # self.U_a -= learning_rate * gradient_clip(dUa, 1)
        # self.U_i -= learning_rate * gradient_clip(dUi, 1)
        # self.U_f -= learning_rate * gradient_clip(dUf, 1)
        # self.U_o -= learning_rate * gradient_clip(dUo, 1)

        self.B_a -= learning_rate * dBa
        self.B_i -= learning_rate * dBi
        self.B_f -= learning_rate * dBf
        self.B_o -= learning_rate * dBo

        return error


if __name__ == '__main__':
    dataset = pd.read_csv('../res/imdb.csv')
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

    x = x.reshape(x.shape[0], x.shape[1], 1)[:128] / np.max(x)
    x = x * np.random.uniform(-1, 1, (16,))
    y = y.reshape(y.shape[0], 1)[:128]

    # x = np.array([
    #     [[1], [2]],
    #     [[1], [1]],
    #     [[0], [1]],
    #     [[2], [2]]
    # ]) * np.random.uniform(-1, 1, (8,))
    # y = np.array([[1], [0], [1], [0]])

    model = Sequential()
    model.add(LSTM(32, input_shape=(32, 16), return_sequence=False))
    # model.add(LSTM(32, input_shape=(32, 32)))
    model.add(Dense(64, input_shape=(32,)))
    model.add(Tanh())
    model.add(Dense(1, input_shape=(64,)))
    model.add(Sigmoid())

    model.compile(loss=BinaryCrossEntropy(), lr=0.01)

    history = model.fit((x, y), epochs=256, batch_size=2)
    plt.plot(history['loss'], label='loss', color='red')
    plt.plot(history['val_loss'], label='val_loss', color='green', linestyle='--')
    plt.legend()
    plt.show()
    plt.plot(history['accuracy'], label='accuracy', color='blue')
    plt.plot(history['val_accuracy'], label='val_accuracy', color='black', linestyle='--')
    plt.legend()
    plt.show()

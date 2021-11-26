import numpy as np
import pandas as pd

from binary_cross_entropy import BinaryCrossEntropy
from dense import Dense
from mean_squared_error import MeanSquaredError
from tanh import Tanh
from text_vetorization import TextVectorization

np.random.seed(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

x = x.reshape(x.shape[0], x.shape[1], 1)[:32]
x = x / np.max(x)
x = x * np.random.uniform(-1, 1, (4,))
y = y.reshape(y.shape[0], 1)[:32]

# x = np.array([
#     [[1], [2]],
#     [[1], [1]],
#     [[0], [1]],
#     [[2], [2]]
# ]) * np.random.uniform(-1, 1, (8,))
# y = np.array([[1], [0], [1], [0]])

learning_rate = 0.01

loss_f = MeanSquaredError()


class LSTM:
    def __init__(self, unit, input_shape):
        self.unit = unit
        self.time, self.feature = input_shape

        self.Wa = np.random.randn(self.feature, unit)
        self.Wi = np.random.randn(self.feature, unit)
        self.Wf = np.random.randn(self.feature, unit)
        self.Wo = np.random.randn(self.feature, unit)

        self.Ua = np.random.randn(unit, unit)
        self.Ui = np.random.randn(unit, unit)
        self.Uf = np.random.randn(unit, unit)
        self.Uo = np.random.randn(unit, unit)

        self.Ba = np.zeros((1, unit))
        self.Bi = np.zeros((1, unit))
        self.Bf = np.zeros((1, unit))
        self.Bo = np.zeros((1, unit))

    def forward(self, x):
        self.input = x
        self.batch = x.shape[0]

        self.a = np.zeros((self.time, self.batch, self.unit))
        self.i = np.zeros((self.time, self.batch, self.unit))
        self.f = np.zeros((self.time, self.batch, self.unit))
        self.o = np.zeros((self.time, self.batch, self.unit))

        self.state = np.zeros((self.time, self.batch, self.unit))
        self.output = np.zeros((self.time, self.batch, self.unit))

        output_prev = np.zeros((self.batch, self.unit))
        state_prev = np.zeros((self.batch, self.unit))

        for t in range(self.time):
            self.a[t] = np.tanh((x[:, t] @ self.Wa) + (output_prev @ self.Ua) + self.Ba)
            self.i[t] = sigmoid((x[:, t] @ self.Wi) + (output_prev @ self.Ui) + self.Bi)
            self.f[t] = sigmoid((x[:, t] @ self.Wf) + (output_prev @ self.Uf) + self.Bf)
            self.o[t] = sigmoid((x[:, t] @ self.Wo) + (output_prev @ self.Uo) + self.Bo)

            self.state[t] = (self.a[t] * self.i[t]) + (self.f[t] * state_prev)
            self.output[t] = np.tanh(self.state[t]) * self.o[t]

            state_prev = self.state[t]
            output_prev = self.output[t]

        return self.output[-1]

    def backward(self, d_loss):
        dWa = np.zeros((self.feature, self.unit))
        dWi = np.zeros((self.feature, self.unit))
        dWf = np.zeros((self.feature, self.unit))
        dWo = np.zeros((self.feature, self.unit))

        dUa = np.zeros((self.unit, self.unit))
        dUi = np.zeros((self.unit, self.unit))
        dUf = np.zeros((self.unit, self.unit))
        dUo = np.zeros((self.unit, self.unit))

        dBa = np.zeros((1, self.unit))
        dBi = np.zeros((1, self.unit))
        dBf = np.zeros((1, self.unit))
        dBo = np.zeros((1, self.unit))

        d_output_next = np.zeros((self.batch, self.unit))
        d_state_next = np.zeros((self.batch, self.unit))
        next_f = np.zeros((self.batch, self.unit))

        for t in reversed(range(self.time)):
            if t == 0:
                output_prev = np.zeros((self.batch, self.unit))
                state_prev = np.zeros((self.batch, self.unit))
            else:
                output_prev = self.output[t - 1]
                state_prev = self.state[t - 1]

            d_out = d_loss + d_output_next
            d_state = d_out * self.o[t] * (1 - np.tanh(self.state[t]) ** 2) + (d_state_next * next_f)

            d_a = d_state * self.i[t] * (1 - self.a[t] ** 2)
            d_i = d_state * self.a[t] * self.i[t] * (1 - self.i[t])
            d_f = d_state * state_prev * self.f[t] * (1 - self.f[t])
            d_o = d_out * np.tanh(self.state[t]) * self.o[t] * (1 - self.o[t])

            dWa += (d_a.T @ self.input[:, t]).T
            dWi += (d_i.T @ self.input[:, t]).T
            dWf += (d_f.T @ self.input[:, t]).T
            dWo += (d_o.T @ self.input[:, t]).T

            dUa += np.sum(d_a * output_prev, axis=0, keepdims=True).T
            dUi += np.sum(d_i * output_prev, axis=0, keepdims=True).T
            dUf += np.sum(d_f * output_prev, axis=0, keepdims=True).T
            dUo += np.sum(d_o * output_prev, axis=0, keepdims=True).T

            dBa += np.sum(d_a, axis=0)
            dBi += np.sum(d_i, axis=0)
            dBf += np.sum(d_f, axis=0)
            dBo += np.sum(d_o, axis=0)

            next_f = self.f[t]
            d_state_next = d_state
            d_output_next = (d_a @ self.Ua) + (d_i @ self.Ui) + (d_f @ self.Uf) + (d_o @ self.Uo)

        self.Wa -= dWa * learning_rate
        self.Wi -= dWi * learning_rate
        self.Wf -= dWf * learning_rate
        self.Wo -= dWo * learning_rate

        self.Ua -= dUa * learning_rate
        self.Ui -= dUi * learning_rate
        self.Uf -= dUf * learning_rate
        self.Uo -= dUo * learning_rate

        self.Ba -= dBa * learning_rate
        self.Bi -= dBi * learning_rate
        self.Bf -= dBf * learning_rate
        self.Bo -= dBo * learning_rate

        return d_output_next


if __name__ == '__main__':
    model = LSTM(8, (32, 4))
    dense_1 = Dense(32, (8,))
    tanh_1 = Tanh()
    dense_2 = Dense(1, (32,))

    for e in range(128):
        for _x, _y in zip(x, y):
            _x, _y = np.array([_x]), np.array([_y])

            logit = model.forward(_x)
            logit = dense_1.forward(logit)
            logit = tanh_1.forward(logit)
            logit = dense_2.forward(logit)

            d_loss = logit - _y

            err = dense_2.backward(d_loss, 0.01)
            err = tanh_1.backward(err, 0.01)
            err = dense_1.backward(err, 0.01)
            err = model.backward(err)
            print(f'{logit[0][0]:0.4f}', _y[0][0])

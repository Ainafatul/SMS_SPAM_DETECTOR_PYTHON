import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from res.activation.Sigmoid import Sigmoid
from res.activation.Tanh import Tanh
from res.layer.Dense import Dense
from res.loss.MeanSquaredError import MeanSquaredError
from res.model.Model import Model
from utils.TextVectorization import TextVectorization


class LSTM:
    # parameters
    # input_size: size of input shape (batch, time_frame, features)
    def __init__(self, units, input_shape):
        self.units = units
        self.batch_size, self.time_frame, self.features = input_shape

        self.Wxf = np.random.randn(self.features, units)
        self.Wxi = np.random.randn(self.features, units)
        self.Wxo = np.random.randn(self.features, units)
        self.Wxc = np.random.randn(self.features, units)

        self.Whf = np.random.randn(units, units)
        self.Whi = np.random.randn(units, units)
        self.Who = np.random.randn(units, units)
        self.Whc = np.random.randn(units, units)

        self.Bf = np.random.randn(units)
        self.Bi = np.random.randn(units)
        self.Bo = np.random.randn(units)
        self.Bc = np.random.randn(units)

        self.f = np.zeros((self.time_frame, self.batch_size, self.units))
        self.i = np.zeros((self.time_frame, self.batch_size, self.units))
        self.o = np.zeros((self.time_frame, self.batch_size, self.units))

        self.state_bar = np.zeros((self.time_frame, self.batch_size, self.units))

        self.state = np.zeros((self.time_frame, self.batch_size, self.units))
        self.output = np.zeros((self.time_frame, self.batch_size, self.units))

    # param: x: (batch_size, time_frame, features)
    def forward(self, x, return_sequence=False):
        output_prev = np.zeros((self.batch_size, self.units))
        state_prev = np.zeros((self.batch_size, self.units))
        for t in range(self.time_frame):
            x_t = x[:, t, :]
            self.f[t] = self.sigmoid(np.dot(x_t, self.Wxf) + np.dot(output_prev, self.Whf))
            self.i[t] = self.sigmoid(np.dot(x_t, self.Wxi) + np.dot(output_prev, self.Whi))
            self.o[t] = self.sigmoid(np.dot(x_t, self.Wxo) + np.dot(output_prev, self.Who))

            self.state_bar[t] = self.tanh(np.dot(x_t, self.Wxc) + np.dot(output_prev, self.Whc))
            self.state[t] = self.f[t] * state_prev + self.i[t] * self.state_bar[t]
            self.output[t] = self.o[t] * self.tanh(self.state[t])

            state_prev = self.state[t]
            output_prev = self.output[t]
        if return_sequence:
            return self.output
        else:
            return self.output[-1]

    # function to compute backward pass of LSTM
    def backward(self, d_loss):
        d_output = np.zeros_like(self.output[0])
        next_state = np.zeros_like(self.state[0])
        next_f = np.zeros_like(self.f[0])
        for t in range(self.time_frame - 1, -1, -1):
            delta_output = d_loss + d_output

            delta_state = delta_output * self.o[t] * self.tanh_prime(self.state[t]) + next_state * next_f
            delta_state_bart = delta_state * self.i[t] * (1 - self.state_bar[t] ** 2)
            delta_it = delta_state * delta_state_bart * self.i[t] * (1 - self.i[t])
            delta_ft = delta_state * self.state[t - 1] * self.f[t] * (1 - self.f[t])
            delta_ot = delta_output * self.tanh(self.state[t]) * self.o[t] * (1 - self.o[t])

            # print(f"delta_it: {delta_it.shape}")
            # print(f"delta_ft: {delta_ft.shape}")
            # print(f"delta_ot: {delta_ot.shape}")
            # print(f"delta_state_bart: {delta_state_bart.shape}")
            #
            # print(f"Wxf: {self.Wxf.shape}")
            # print(f"Wxi: {self.Wxi.shape}")
            # print(f"Wxo: {self.Wxo.shape}")
            # print(f"Wxc: {self.Wxc.shape}")

            self.Whf += np.mean(delta_ft) * .001
            self.Whi += np.mean(delta_it) * .001
            self.Who += np.mean(delta_ot) * .001
            self.Whc += np.mean(delta_state_bart) * .001

            self.Wxf += np.mean(delta_ft) * .001
            self.Wxi += np.mean(delta_it) * .001
            self.Wxo += np.mean(delta_ot) * .001
            self.Wxc += np.mean(delta_state_bart) * .001

            next_f = self.f[t]
            next_state = self.state[t]

    def update(self, learning_rate):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1 - np.tanh(x) ** 2


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def mse_prime(y_pred, y_true):
    return 2 * (y_pred - y_true) / len(y_true)


if __name__ == '__main__':
    dataset = pd.read_csv('../../dataset/imdb.csv')
    x = dataset.iloc[:, 0].values
    y = dataset.iloc[:, -1].values

    encoder = TextVectorization(1024, 32)
    encoder.fit(x)

    # Input preprocessing
    x = encoder(x)
    # Convert Label from ['negative','positive'] to binary representation
    y = np.array([1 if label == 'positive' else 0 for label in y])

    # Shuffle x and y with the same indices
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    x = x.reshape(x.shape[0], x.shape[1], 1)
    y = y.reshape(y.shape[0], 1)

    model = Model(MeanSquaredError())
    model.add(LSTM(64, input_shape=x.shape))
    model.add(Dense(64, 64))
    model.add(Dense(64, 1))
    model.add(Sigmoid())

    history = model.fit(x, y, batch_size=999, epochs=128, learning_rate=.001)

    plt.plot(history)
    plt.show()

    test = np.array([
        [encoder.transform("this is a good movie")],
    ])
    print(f"test :{test.shape}")
    test = np.reshape(test, (1, 32, 1))
    print(f"test :{test.shape}")
    result = model.predict(test)
    print(f"Result: {np.mean(result)}")

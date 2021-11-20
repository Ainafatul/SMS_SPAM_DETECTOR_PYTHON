import numpy as np


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
    def backward(self, x, d_loss):
        delta_ot = np.zeros_like(self.output[0])
        d_output = np.zeros_like(self.output[0])
        f_next = np.zeros_like(self.f[0])
        for t in range(self.time_frame - 1, -1, -1):
            delta_output = d_loss + d_output
            delta_state = delta_output * self.o[t] * self.tanh_prime(self.state[t])
            delta_state_bart = delta_state * self.i[t] * (1 - self.state_bar[t] ** 2)
            delta_it = delta_state * delta_state_bart * self.i[t] * (1 - self.i[t])
            delta_ft = delta_state * self.state[t - 1] * self.f[t] * (1 - self.f[t])
            delta_ot = delta_output * self.tanh(self.state[t]) * self.o[t] * (1 - self.o[t])

            delta_x = self.Wxc

            delta_c_next = delta_output_bart
            d_output = np.dot(delta_output, self.Whf.T) + np.dot(delta_ot, self.Who.T) + np.dot(delta_ct, self.Whc.T)
            f_next = self.f[t]
            exit()

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
    n = 1
    features = 32
    batch = 4
    time = 12
    x = np.random.random(size=(batch, time, features))

    cell = LSTM(units=n, input_shape=x.shape)

    h = np.random.randn(batch, n)
    c = np.random.randn(batch, n)

    h = cell.forward(x)
    dloss = mse_prime(h, np.ones((batch, n)))
    print(f"h : {h}")
    print(f"loss : {dloss}")
    cell.backward(x, dloss)

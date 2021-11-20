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

        self.h = np.zeros((self.batch_size, self.time_frame, self.units))

    # param: x: (batch_size, time_frame, features)
    def forward(self, x, return_sequence=False):
        h_prev = np.zeros((self.batch_size, self.units))
        c_prev = np.zeros((self.batch_size, self.units))
        for t in range(self.time_frame):
            x_t = x[:, t, :]
            f = self.sigmoid(np.dot(x_t, self.Wxf) + np.dot(h_prev, self.Whf))
            i = self.sigmoid(np.dot(x_t, self.Wxi) + np.dot(h_prev, self.Whi))
            o = self.sigmoid(np.dot(x_t, self.Wxo) + np.dot(h_prev, self.Who))
            c_t = self.tanh(np.dot(x_t, self.Wxc) + np.dot(h_prev, self.Whc))
            c_prev = f * c_prev + i * c_t
            h_prev = o * self.tanh(c_prev)
            self.h[:, t, :] = h_prev
        if return_sequence:
            return self.h
        else:
            return self.h[:, -1, :]

    # function to compute backward pass of LSTM
    def backward(self, x, y):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)


if __name__ == '__main__':
    n = 8
    features = 32
    batch = 4
    time = 12
    x = np.random.random(size=(batch, time, features))

    cell = LSTM(units=n, input_shape=x.shape)

    h = np.random.randn(batch, n)
    c = np.random.randn(batch, n)

    h = cell.forward(x)

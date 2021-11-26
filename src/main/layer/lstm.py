import numpy as np


def sigmoid(x):
    np.clip(x, -500, 500, out=x)
    return 1 / (1 + np.exp(-x))


class LSTM:
    def __init__(self, unit, input_shape, return_sequences=False, backprop_sequence=False):
        self.unit = unit
        self.time, self.feature = input_shape
        self.return_sequences = return_sequences
        self.backprop_sequence = backprop_sequence

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

        if self.return_sequences:
            return np.moveaxis(self.output, 0, 1)
        return self.output[-1]

    def backward(self, d_loss, learning_rate):
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

        d_output = d_loss
        d_outputs = np.zeros((self.batch, self.time, self.unit))
        for t in reversed(range(self.time)):
            if t == 0:
                output_prev = np.zeros((self.batch, self.unit))
                state_prev = np.zeros((self.batch, self.unit))
            else:
                output_prev = self.output[t - 1]
                state_prev = self.state[t - 1]

            if self.return_sequences:
                d_out = d_outputs[:, t]
            else:
                d_out = d_output + d_output_next
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

            d_output = np.zeros_like(d_output)
            next_f = self.f[t]
            d_state_next = d_state
            d_output_next = (d_a @ self.Ua) + (d_i @ self.Ui) + (d_f @ self.Uf) + (d_o @ self.Uo)
            d_outputs[:, t] = d_output_next

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

        if self.backprop_sequence:
            return d_outputs
        return d_output_next

import numpy as np

from binary_cross_entropy import BinaryCrossEntropy
from mean_squared_error import MeanSquaredError

np.random.seed(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


batch = 4
unit = 1
time = 2
feature = 2

x = np.array([
    [[1, 2], [.5, 3]],
    [[2, 2], [.5, 3]],
    [[1, 1], [.5, 3]],
    [[2, 1], [.5, 3]],
])
y = np.array([
    [1],
    [0],
    [0],
    [1],
]
)

Wa = np.random.randn(feature, unit)
Wi = np.random.randn(feature, unit)
Wf = np.random.randn(feature, unit)
Wo = np.random.randn(feature, unit)

Ua = np.random.randn(unit, unit)
Ui = np.random.randn(unit, unit)
Uf = np.random.randn(unit, unit)
Uo = np.random.randn(unit, unit)

Ba = np.zeros((1, unit))
Bi = np.zeros((1, unit))
Bf = np.zeros((1, unit))
Bo = np.zeros((1, unit))

# Wa = np.array([[.45], [.25]])
# Wi = np.array([[.95], [.8]])
# Wf = np.array([[.7], [.45]])
# Wo = np.array([[.6], [.4]])
#
# Ua = np.array([[.15]])
# Ui = np.array([[.8]])
# Uf = np.array([[.1]])
# Uo = np.array([[.25]])
#
# Ba = np.array([[.2]])
# Bi = np.array([[.65]])
# Bf = np.array([[.15]])
# Bo = np.array([[.1]])

learning_rate = 0.01

loss_f = MeanSquaredError()


class LSTM:
    def __init__(self, unit, input_shape):
        self.unit = unit
        self.time, self.feature = input_shape

        self.Wa = np.random.randn(feature, unit)
        self.Wi = np.random.randn(feature, unit)
        self.Wf = np.random.randn(feature, unit)
        self.Wo = np.random.randn(feature, unit)

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
        self.a = np.zeros((time, batch, unit))
        self.i = np.zeros((time, batch, unit))
        self.f = np.zeros((time, batch, unit))
        self.o = np.zeros((time, batch, unit))

        self.state = np.zeros((time, batch, unit))
        self.output = np.zeros((time, batch, unit))

        output_prev = np.zeros((batch, unit))
        state_prev = np.zeros((batch, unit))

        for t in range(time):
            self.a[t] = np.tanh((x[:, t] @ Wa) + (output_prev @ Ua) + Ba)
            self.i[t] = sigmoid((x[:, t] @ Wi) + (output_prev @ Ui) + Bi)
            self.f[t] = sigmoid((x[:, t] @ Wf) + (output_prev @ Uf) + Bf)
            self.o[t] = sigmoid((x[:, t] @ Wo) + (output_prev @ Uo) + Bo)

            self.state[t] = (self.a[t] * self.i[t]) + (self.f[t] * state_prev)
            self.output[t] = np.tanh(self.state[t]) * self.o[t]

            state_prev = self.state[t]
            output_prev = self.output[t]

        return output[-1]

    def backward(self, d_loss):
        dWa = np.zeros((feature, unit))
        dWi = np.zeros((feature, unit))
        dWf = np.zeros((feature, unit))
        dWo = np.zeros((feature, unit))

        dUa = np.zeros((unit, unit))
        dUi = np.zeros((unit, unit))
        dUf = np.zeros((unit, unit))
        dUo = np.zeros((unit, unit))

        dBa = np.zeros((1, unit))
        dBi = np.zeros((1, unit))
        dBf = np.zeros((1, unit))
        dBo = np.zeros((1, unit))

        d_output_next = np.zeros((batch, unit))
        d_state_next = np.zeros((batch, unit))
        next_f = np.zeros((batch, unit))

        for t in reversed(range(time)):
            if t == 0:
                output_prev = np.zeros((batch, unit))
                state_prev = np.zeros((batch, unit))
            else:
                output_prev = output[t - 1]
                state_prev = state[t - 1]

            d_out = loss + d_output_next
            d_state = d_out * self.o[t] * (1 - np.tanh(self.state[t]) ** 2) + (d_state_next * next_f)

            d_a = d_state * self.i[t] * (1 - self.a[t] ** 2)
            d_i = d_state * self.a[t] * self.i[t] * (1 - self.i[t])
            d_f = d_state * state_prev * self.f[t] * (1 - self.f[t])
            d_o = d_out * np.tanh(self.state[t]) * self.o[t] * (1 - self.o[t])

            dWa += np.sum(d_a * self.input[:, t], axis=0, keepdims=True).T
            dWi += np.sum(d_i * self.input[:, t], axis=0, keepdims=True).T
            dWf += np.sum(d_f * self.input[:, t], axis=0, keepdims=True).T
            dWo += np.sum(d_o * self.input[:, t], axis=0, keepdims=True).T

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


if __name__ == '__main__':

    for e in range(128):
        a = np.zeros((time, batch, unit))
        i = np.zeros((time, batch, unit))
        f = np.zeros((time, batch, unit))
        o = np.zeros((time, batch, unit))
        state = np.zeros((time, batch, unit))
        output = np.zeros((time, batch, unit))

        output_prev = np.zeros((batch, unit))
        state_prev = np.zeros((batch, unit))

        for t in range(time):
            print('t:', x[:, t].shape)
            print('Wa:', Wa.shape)

            a[t] = np.tanh((x[:, t] @ Wa) + (output_prev @ Ua) + Ba)
            i[t] = sigmoid((x[:, t] @ Wi) + (output_prev @ Ui) + Bi)
            f[t] = sigmoid((x[:, t] @ Wf) + (output_prev @ Uf) + Bf)
            o[t] = sigmoid((x[:, t] @ Wo) + (output_prev @ Uo) + Bo)

            state[t] = (a[t] * i[t]) + (f[t] * state_prev)
            output[t] = np.tanh(state[t]) * o[t]

            state_prev = state[t]
            output_prev = output[t]

        loss = output[-1] - y
        print('Loss - ', loss_f.calc(output[-1], y))

        dWa = np.zeros((feature, unit))
        dWi = np.zeros((feature, unit))
        dWf = np.zeros((feature, unit))
        dWo = np.zeros((feature, unit))

        dUa = np.zeros((unit, unit))
        dUi = np.zeros((unit, unit))
        dUf = np.zeros((unit, unit))
        dUo = np.zeros((unit, unit))

        dBa = np.zeros((1, unit))
        dBi = np.zeros((1, unit))
        dBf = np.zeros((1, unit))
        dBo = np.zeros((1, unit))

        d_output_next = np.zeros((batch, unit))
        d_state_next = np.zeros((batch, unit))
        next_f = np.zeros((batch, unit))

        for t in reversed(range(time)):
            if t == 0:
                output_prev = np.zeros((batch, unit))
                state_prev = np.zeros((batch, unit))
            else:
                output_prev = output[t - 1]
                state_prev = state[t - 1]

            d_out = loss + d_output_next
            d_state = d_out * o[t] * (1 - np.tanh(state[t]) ** 2) + (d_state_next * next_f)

            d_a = d_state * i[t] * (1 - a[t] ** 2)
            d_i = d_state * a[t] * i[t] * (1 - i[t])
            d_f = d_state * state_prev * f[t] * (1 - f[t])
            d_o = d_out * np.tanh(state[t]) * o[t] * (1 - o[t])

            dWa += np.sum(d_a * x[:, t], axis=0, keepdims=True).T
            dWi += np.sum(d_i * x[:, t], axis=0, keepdims=True).T
            dWf += np.sum(d_f * x[:, t], axis=0, keepdims=True).T
            dWo += np.sum(d_o * x[:, t], axis=0, keepdims=True).T

            dUa += np.sum(d_a * output_prev, axis=0, keepdims=True).T
            dUi += np.sum(d_i * output_prev, axis=0, keepdims=True).T
            dUf += np.sum(d_f * output_prev, axis=0, keepdims=True).T
            dUo += np.sum(d_o * output_prev, axis=0, keepdims=True).T

            dBa += np.sum(d_a, axis=0)
            dBi += np.sum(d_i, axis=0)
            dBf += np.sum(d_f, axis=0)
            dBo += np.sum(d_o, axis=0)

            next_f = f[t]
            d_state_next = d_state
            d_output_next = (d_a @ Ua) + (d_i @ Ui) + (d_f @ Uf) + (d_o @ Uo)

        Wa -= dWa * learning_rate
        Wi -= dWi * learning_rate
        Wf -= dWf * learning_rate
        Wo -= dWo * learning_rate

        Ua -= dUa * learning_rate
        Ui -= dUi * learning_rate
        Uf -= dUf * learning_rate
        Uo -= dUo * learning_rate

        Ba -= dBa * learning_rate
        Bi -= dBi * learning_rate
        Bf -= dBf * learning_rate
        Bo -= dBo * learning_rate

    print(output[-1])

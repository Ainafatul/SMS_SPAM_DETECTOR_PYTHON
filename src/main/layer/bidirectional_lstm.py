import numpy as np

from lstm import LSTM


class BidirectionalLSTM:

    def __init__(self, unit, input_shape):
        self.unit = unit
        self.layer_forward = LSTM(unit, input_shape)
        self.layer_backward = LSTM(unit, input_shape)

    def forward(self, x):
        self.input_forward = x
        self.input_reserve = np.flip(x, axis=1)

        forward = self.layer_forward.forward(self.input_forward)
        backward = self.layer_backward.forward(self.input_reserve)
        return np.concatenate((forward, backward), axis=1)

    def backward(self, d_loss, learning_rate):
        d_loss_forward = d_loss[:, :self.unit]
        d_loss_reserve = d_loss[:, self.unit:]

        self.layer_forward.backward(d_loss_forward, learning_rate)
        self.layer_backward.backward(d_loss_reserve, learning_rate)

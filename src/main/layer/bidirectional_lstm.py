import numpy as np

from lstm import LSTM


class Bidirectional_LSTM:

    def __init__(self, unit, input_shape, return_sequences=False, backprop_sequence=False):
        self.unit = unit
        self.layer_forward = LSTM(unit, input_shape, return_sequences=return_sequences, backprop_sequence=backprop_sequence)
        self.layer_backward = LSTM(unit, input_shape, return_sequences=return_sequences, backprop_sequence=backprop_sequence)

    def forward(self, x, training):
        self.input_forward = x
        self.input_reserve = np.flip(x, axis=1)
        forward = self.layer_forward.forward(self.input_forward, training)
        backward = self.layer_backward.forward(self.input_reserve, training)
        return np.concatenate((forward, backward), axis=-1)

    def backward(self, d_loss, learning_rate):
        d_loss_forward = d_loss[:, :self.unit]
        d_loss_reserve = d_loss[:, self.unit:]

        f = self.layer_forward.backward(d_loss_forward, learning_rate)
        b = self.layer_backward.backward(d_loss_reserve, learning_rate)
        return np.concatenate((f, b), axis=-1)

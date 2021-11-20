import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from res.activation.Sigmoid import Sigmoid
from res.activation.Tanh import Tanh
from res.layer.Dense import Dense
from res.layer.Layer import Layer
from res.loss.MeanSquaredError import MeanSquaredError
from res.model.Model import Model
from utils.TextVectorization import TextVectorization


class LSTMCell:
    def __init__(self, units, features):
        self.Wf = np.random.randn(units, features)
        self.Wi = np.random.randn(units, features)
        self.Wc = np.random.randn(units, features)
        self.Wo = np.random.randn(units, features)

        self.bf = np.zeros(units)
        self.bi = np.zeros(units)
        self.bc = np.zeros(units)
        self.bo = np.zeros(units)

        self.Uf = np.random.randn(units, units)
        self.Ui = np.random.randn(units, units)
        self.Uc = np.random.randn(units, units)
        self.Uo = np.random.randn(units, units)

    def forget_gate(self, x, h):
        return self.sigmoid(self.Wf* x + self.Uf* h + self.bf)

    def input_gate(self, x, h):
        return self.sigmoid(self.Wi * x + self.Ui * h + self.bi)

    def cell_gate(self, c_prev, f, i, c):
        return f * c_prev + i * c

    def output_gate(self, x, h):
        return self.sigmoid(self.Wo * x + self.Uo * h + self.bo)

    def candidate_gate(self, x, h):
        return self.tanh(self.Wc * x + self.Uc * h + self.bc)

    def forward(self, X, h_prev, c_prev):
        self.logit = X
        print(f"logit : {X.shape}")
        print(f"h_prev : {h_prev.shape}")
        print(f"c_prev : {c_prev.shape}")
        print(f"Wf : {self.Wf.shape}")
        f = self.forget_gate(X, h_prev)
        i = self.input_gate(X, h_prev)
        c = self.candidate_gate(X, h_prev)
        o = self.output_gate(X, h_prev)
        c_next = self.cell_gate(c_prev, f, i, c)
        h_next = self.tanh(c_next) * o
        print("f:", f.shape)
        print("i:", i.shape)
        print("c:", c.shape)
        print("o:", o.shape)
        print("c_next:", c_next.shape)
        print("h_next:", h_next.shape)

        return h_next, c_next

    # function to compute derivative weights and biases
    # return : d_h_prev, d_c_prev
    def backward(self, d_next, c_next, c_prev, h_prev):
        do = self.tanh(c_next) * d_next
        dc = self.output_gate(self.logit, h_prev) * d_next + (self.Uo.T * do)
        di = self.input_gate(self.logit, h_prev) * dc
        df = self.forget_gate(self.logit, h_prev) * dc
        dg = self.candidate_gate(self.logit, h_prev) * dc
        self.dWf = np.outer(df, self.logit)
        self.dWi = np.outer(di, self.logit)
        self.dWc = np.outer(dg, self.logit)
        self.dWo = np.outer(do, self.logit)
        self.dbf = df
        self.dbi = di
        self.dbc = dc
        self.dbo = do
        self.dUf = np.outer(df, h_prev)
        self.dUi = np.outer(di, h_prev)
        self.dUc = np.outer(dg, h_prev)
        self.dUo = np.outer(do, h_prev)
        d_h_prev = self.Uo.T * do
        d_c_prev = self.Uc.T * dg
        return d_h_prev, d_c_prev

    # function to update the weights and biases
    def update(self, learning_rate):
        self.Wf -= learning_rate * self.dWf
        self.Wi -= learning_rate * self.dWi
        self.Wc -= learning_rate * self.dWc
        self.Wo -= learning_rate * self.dWo
        self.bf -= learning_rate * self.dbf
        self.bi -= learning_rate * self.dbi
        self.bc -= learning_rate * self.dbc
        self.bo -= learning_rate * self.dbo
        self.Uf -= learning_rate * self.dUf
        self.Ui -= learning_rate * self.dUi
        self.Uc -= learning_rate * self.dUc
        self.Uo -= learning_rate * self.dUo

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)


class LSTM(Layer):
    def __init__(self, units, features):
        self.units = units
        self.features = features
        self.cell = LSTMCell(units, features)

    def forward(self, X, return_sequence=False):
        # print(X.shape)
        # exit()
        self.h_prev = np.zeros(self.features)
        self.c_prev = np.zeros(self.features)
        self.h = np.zeros((self.units, self.features))
        self.c = np.zeros((self.units, self.features))
        for i in range(self.units):
            self.h[i], self.c[i] = self.cell.forward(X[i], self.h_prev, self.c_prev)
            self.h_prev = self.h[i]
            self.c_prev = self.c[i]
        if return_sequence:
            return self.h
        else:
            return np.expand_dims(self.h[-1], axis=0)

    def backward(self, d_next, return_sequence=False):
        d_h_prev = np.zeros(self.units)
        d_c_prev = np.zeros(self.units)
        d_h = np.zeros((d_next.shape[1], self.units, self.units))
        d_c = np.zeros((d_next.shape[1], self.units, self.units))
        for i in range(d_next.shape[1]):
            d_h[i], d_c[i] = self.cell.backward(d_next[i], self.c[i], self.h_prev, self.c_prev)
            d_h_prev = d_h[i]
            d_c_prev = d_c[i]
        if return_sequence:
            return d_h
        else:
            return np.expand_dims(d_h[-1], axis=0)

    def update(self, learning_rate):
        self.cell.update(learning_rate)


if __name__ == '__main__':
    # features = 2
    # time = 8
    # units = 8
    # lstm = LSTM(units=units, features=features)
    # X = np.random.randn(time, features)
    # h_prev = np.random.randn(units)
    # c_prev = np.random.randn(units)
    # h = lstm.forward(X)

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
    print(x.shape)
    print(y.shape)

    model = Model(MeanSquaredError())
    model.add(LSTM(8, features=1))
    model.add(Tanh())
    model.add(Dense(8, 1))
    model.add(Sigmoid())

    history = model.fit(x, y, epochs=128, learning_rate=.02)

    plt.plot(history)
    plt.show()

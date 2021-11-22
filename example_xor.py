import matplotlib.pyplot as plt
import numpy as np

from res.activation.Linear import Linear
from res.activation.ReLU import ReLU
from res.layer.Dense import Dense
from res.layer.Dropout import Dropout
from res.loss.BinaryCrossEntropy import BinaryCrossEntropy
from res.model.Model import Model
from res.activation.Sigmoid import Sigmoid
from res.activation.Tanh import Tanh
from res.loss.MeanSquaredError import MeanSquaredError

if __name__ == '__main__':
    size = 1024
    x_train = np.random.randint(0, 2, size=(size, 2))
    y_train = np.logical_xor(x_train[:, 0], x_train[:, 1])
    x_train = np.reshape(x_train, (size, 2))
    y_train = np.reshape(y_train, (size, 1))

    model = Model(MeanSquaredError())
    model.add(Dense(2, 8))
    model.add(Tanh())
    model.add(Dense(8, 1))
    model.add(Sigmoid())

    history = model.fit(x_train, y_train, batch_size=64, epochs=32, learning_rate=.01)

    plt.plot(history)
    plt.show()

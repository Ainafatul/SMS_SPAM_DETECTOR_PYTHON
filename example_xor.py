import matplotlib.pyplot as plt
import numpy as np

from res.layer.Dense import Dense
from res.model.Model import Model
from res.activation.Sigmoid import Sigmoid
from res.activation.Tanh import Tanh
from res.layer.Dropout import Dropout
from res.loss.MeanSquaredError import MeanSquaredError

if __name__ == '__main__':
    x_train = np.random.randint(0, 2, size=(512, 2))
    y_train = np.logical_xor(x_train[:, 0], x_train[:, 1])
    x_train = np.reshape(x_train, (512, 2))
    y_train = np.reshape(y_train, (512, 1))

    model = Model(MeanSquaredError())
    model.add(Dense(2, 32))
    model.add(Tanh())
    model.add(Dense(32, 1))
    model.add(Sigmoid())

    history = model.fit(x_train, y_train, epochs=32, learning_rate=0.01)
    plt.plot(history)
    plt.show()


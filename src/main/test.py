import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Sequential import Sequential
from dense import Dense
from sigmoid import Sigmoid
from tanh import Tanh
from text_vetorization import TextVectorization


def gradient_descent(gradient, start=0, learn_rate=.001, n_iter=50, tolerance=1e-06):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector


if __name__ == '__main__':
    dataset = pd.read_csv('../res/imdb.csv')
    x = dataset.iloc[:, 0].values
    y = dataset.iloc[:, -1].values

    encoder = TextVectorization(1024, 32)
    encoder.fit(x)

    x = encoder(x)
    y = np.array([1 if label == 'positive' else 0 for label in y])

    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    x = x.reshape(x.shape[0], x.shape[1])
    y = y.reshape(y.shape[0], 1)

    net = Sequential()
    net.add(Dense(32, 64))
    net.add(Tanh())
    net.add(Dense(64, 8))
    net.add(Tanh())
    net.add(Dense(8, 1))
    net.add(Sigmoid())

    loss = net.fit(x, y, epochs=128, learning_rate=0.01)

    plt.plot(loss)
    plt.show()

    out = net.predict(x[:4])
    print(y[:4])
    print(out > .5)
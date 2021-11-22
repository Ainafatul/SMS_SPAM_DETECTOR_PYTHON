import re

import matplotlib.pyplot as plt
import numpy as np

from res.layer.Dense import Dense
from res.loss.BinaryCrossEntropy import BinaryCrossEntropy
from res.model.Model import Model
from res.activation.Sigmoid import Sigmoid
from res.activation.Tanh import Tanh
from res.loss.MeanSquaredError import MeanSquaredError

import pandas as pd

from utils.TextVectorization import TextVectorization

if __name__ == '__main__':
    dataset = pd.read_csv('dataset/imdb.csv')
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

    x = x.reshape(x.shape[0], x.shape[1])
    y = y.reshape(y.shape[0], 1)

    model = Model(MeanSquaredError())
    model.add(Dense(32, 64))
    model.add(Tanh())
    model.add(Dense(64, 32))
    model.add(Tanh())
    model.add(Dense(32, 8))
    model.add(Tanh())
    model.add(Dense(8, 1))
    model.add(Sigmoid())

    history = model.fit(x, y, batch_size=32, epochs=32, learning_rate=.01)

    plt.plot(history)
    plt.show()

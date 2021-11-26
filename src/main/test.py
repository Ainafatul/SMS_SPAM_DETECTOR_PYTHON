import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Sequential import Sequential
from binary_cross_entropy import BinaryCrossEntropy
from dense import Dense
from sigmoid import Sigmoid
from tanh import Tanh
from text_vetorization import TextVectorization

if __name__ == '__main__':
    dataset = pd.read_csv('../res/imdb.csv')
    x = dataset.iloc[:, 0].values
    y = dataset.iloc[:, -1].values

    encoder = TextVectorization(1024, 32)
    encoder.fit(x)

    x = encoder(x)
    y = np.array([1 if label == 'positive' else 0 for label in y])

    x = x.reshape(x.shape[0], x.shape[1])
    y = y.reshape(y.shape[0], 1)

    train = .8
    x_train, y_train = x[:int(len(x) * train)], y[:int(len(y) * train)]
    x_val, y_val = x[int(len(x) * train):], y[int(len(y) * train):]

    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    model = Sequential()
    model.add(Dense(64, input_shape=(32,)))
    model.add(Tanh())
    model.add(Dense(32, input_shape=(64,)))
    model.add(Tanh())
    model.add(Dense(1, input_shape=(32,)))
    model.add(Sigmoid())

    model.compile(loss=BinaryCrossEntropy(), lr=.001)

    history = model.fit((x_train, y_train), val=(x_val, y_val), batch_size=32, epochs=1024, decay=0.001)

    plt.plot(history['loss'], label='loss', color='red')
    plt.plot(history['val_loss'], label='val_loss', color='green', linestyle='--')
    plt.legend()
    plt.show()
    plt.plot(history['accuracy'], label='accuracy', color='blue')
    plt.plot(history['val_accuracy'], label='val_accuracy', color='black', linestyle='--')
    plt.legend()
    plt.show()

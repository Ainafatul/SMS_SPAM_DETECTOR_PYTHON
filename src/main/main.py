import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bidirectional_lstm import BidirectionalLSTM
from embeding import Embedding
from tanh import Tanh
from dense import Dense
from sigmoid import Sigmoid
from Sequential import Sequential
from text_vetorization import TextVectorization
from binary_cross_entropy import BinaryCrossEntropy

np.random.seed(0)

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

    x = x.reshape(x.shape[0], x.shape[1], 1)
    y = y.reshape(y.shape[0], 1)

    train_size = 64
    val_size = 64

    x_train = x[:train_size]
    y_train = y[:train_size]

    x_test = x[train_size:train_size + val_size]
    y_test = y[train_size:train_size + val_size]

    model = Sequential()
    model.add(Embedding(encoder.vocab_size, 8))
    model.add(BidirectionalLSTM(32, input_shape=(x.shape[1], 8)))
    model.add(Dense(64, input_shape=(64,)))
    model.add(Tanh())
    model.add(Dense(32, input_shape=(64,)))
    model.add(Tanh())
    model.add(Dense(1, input_shape=(32,)))
    model.add(Sigmoid())

    model.compile(loss=BinaryCrossEntropy(), lr=0.00001)

    history = model.fit((x_train, y_train), val=(x_test, y_test), epochs=512, batch_size=16, decay=0.001)
    plt.plot(history['val_loss'], 'b', marker='x', label='val_loss', linestyle='--')
    plt.plot(history['loss'], label='loss', color='red')
    plt.legend()
    plt.show()
    plt.plot(history['val_accuracy'], 'b', marker='x', label='val_accuracy', linestyle='--')
    plt.plot(history['accuracy'], label='accuracy', color='red')
    plt.legend()
    plt.show()

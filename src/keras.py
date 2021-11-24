import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam

from text_vetorization import TextVectorization

if __name__ == '__main__':
    dataset = pd.read_csv('res/imdb.csv')
    x = dataset.iloc[:, 0].values
    y = dataset.iloc[:, -1].values

    encoder = TextVectorization(1024, 32)
    encoder.fit(x)

    x = encoder(x)
    y = np.array([1 if label == 'positive' else 0 for label in y])

    x = np.eye(1024)[x]
    a = np.array([np.eye(x)[x] for x in range(len(y))])
    print(a.shape)
    exit()


    x = x.reshape(x.shape[0], x.shape[1])
    y = y.reshape(y.shape[0], 1)

    train = .8
    x_train, y_train = x[:int(len(x) * train)], y[:int(len(y) * train)]
    x_val, y_val = x[int(len(x) * train):], y[int(len(y) * train):]

    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    model = Sequential([
        LSTM(32, input_shape=(32, 1), return_sequences=True),
        LSTM(1),
    ])

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=.001), metrics=['accuracy'])
    model.summary()

    model.fit(x, y, epochs=32)

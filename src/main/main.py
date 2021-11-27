import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bidirectional_lstm import Bidirectional_LSTM
from dropout import Dropout
from embeding import Embedding
from relu import ReLU
from tanh import Tanh
from dense import Dense
from sigmoid import Sigmoid
from Sequential import Sequential
from text_vetorization import TextVectorization
from binary_cross_entropy import BinaryCrossEntropy

np.random.seed(0)

if __name__ == '__main__':
    # membaca dataset pada file csv
    dataset = pd.read_csv('../res/imdb.csv')

    # : batch, 0 = teks
    x = dataset.iloc[:, 0].values
    #label( -1 = paling akhir pada kolom)
    y = dataset.iloc[:, -1].values

    # inisialisasi prepro dengan max_feature 10240 dan max_seq 64
    encoder = TextVectorization(1024, 32)
    # fit ->dictionary->tokenisasi
    encoder.fit(x)

    # x = input
    x = encoder(x)
    # inisialisasi output
    y = np.array([1 if label == 'positive' else 0 for label in y])

    # inisialisasi indesis untuk mengatur array ke x
    indices = np.arange(len(x))
    # dilakukan pengacakan terhadap urutan sebelumnya
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # (32,1,1)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    y = y.reshape(y.shape[0], 1)

    # membagi jumlah dataset
    train_size = 32
    val_size = 16

    x_train = x[:train_size]
    y_train = y[:train_size]

    #
    x_test = x[train_size:train_size + val_size]
    y_test = y[train_size:train_size + val_size]

    # membuat model
    model = Sequential()
    model.add(Embedding(encoder.vocab_size, 8))
    model.add(Bidirectional_LSTM(32, input_shape=(32, 8), return_sequences=True))
    model.add(Bidirectional_LSTM(16, input_shape=(32, 64), backprop_sequence=True))
    model.add(Dense(32, input_shape=(32,)))
    model.add(ReLU())
    model.add(Dropout(0.125))
    model.add(Dense(1, input_shape=(32,)))
    model.add(Sigmoid())

    model.compile(loss=BinaryCrossEntropy(), lr=0.01)

    #
    history = model.fit((x_train, y_train), val=(x_test, y_test), epochs=128, batch_size=8, decay=0.001)
    plt.plot(history['val_loss'], 'b', marker='x', label='val_loss', linestyle='--')
    plt.plot(history['loss'], label='train_loss', color='red')
    plt.legend()
    plt.show()
    plt.plot(history['val_accuracy'], 'b', marker='x', label='val_accuracy', linestyle='--')
    plt.plot(history['accuracy'], label='train_accuracy', color='red')
    plt.legend()
    plt.show()

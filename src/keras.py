import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM

from text_vetorization import TextVectorization

if __name__ == '__main__':
    lstm = LSTM(units=32, input_shape=(1, 1), return_sequences=False)

    x = np.random.random((1, 32, 32))
    x = lstm(x)

    print(x.shape)

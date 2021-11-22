import os

import numpy as np
import pandas as pd

from res.layer.Dense import Dense
from res.loss.MeanSquaredError import MeanSquaredError
from res.model.Model import Model

if __name__ == '__main__':
    df = pd.read_csv('dataset/iris_training.csv')
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    print(f'Input: {x.shape}')
    print(f'Label: {y.shape}')

    loss = MeanSquaredError()

    model = Model(loss)
    model.add(Dense(8, input_shape=4))
    model.add(Dense(1, input_shape=8))

    model.fit(x, y, epochs=32, batch_size=32, learning_rate=.1)

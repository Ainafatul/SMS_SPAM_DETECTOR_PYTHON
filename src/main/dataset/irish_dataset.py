import numpy as np
import pandas as pd


class IrishDataset:

    def __init__(self):
        df = pd.read_csv('../res/iris_training.csv')
        self.x = np.array(df.iloc[:, :-1].values)
        self.y = np.array(df.iloc[:, -1].values)
        self.y = np.reshape(self.y, (len(self.y), 1))

    def training(self):
        size = len(self.y)
        return self.x[:int(size / .8)], self.y[:int(size / .8)]

    def validation(self):
        size = len(self.y)
        return self.x[int(size / .8):], self.y[int(size / .8):]

    def take(self, n):
        return self.x[:n], self.y[:n]
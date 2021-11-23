import numpy as np


class XORDataset:
    def __init__(self, size):
        self.x = np.random.randint(0, 2, (size, 2))
        self.y = np.where((self.x[:, 0] == self.x[:, 1]) == 1, 1, 0)
        self.y = self.y.reshape(size, 1)

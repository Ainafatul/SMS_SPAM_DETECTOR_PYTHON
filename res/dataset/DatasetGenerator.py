import matplotlib.pyplot as plt
import numpy as np


class DatasetGenerator:

    def __call__(self, size):
        size = round(size / 2)
        x1 = self.create_set(size, 1, .1)
        y1 = np.full((size, 1), 0)
        x2 = self.create_set(size, .5, .1)
        y2 = np.full((size, 1), 1)
        randomize = np.arange(size*2)
        np.random.shuffle(randomize)
        return np.asarray([*x1, *x2])[randomize], np.asarray([*y1, *y2])[randomize]

    def create_set(self, size, radius, deviation):
        radius = np.random.uniform(radius - deviation, radius + deviation, size)
        angle = np.random.uniform(0, np.pi * 2, size)
        return np.asarray([radius * np.cos(angle), radius * np.sin(angle)]).T


if __name__ == '__main__':
    generator = DatasetGenerator()
    x, y = generator(1024)
    print(x.shape, y.shape)
    for i in range(len(x)):
        if y[i] == 1:
            plt.plot(x[i][0], x[i][1], 'bo')
        else:
            plt.plot(x[i][0], x[i][1], 'ro')
    plt.show()

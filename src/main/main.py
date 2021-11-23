import numpy as np

from dense import Dense
from mean_squared_error import MeanSquaredError

mse = MeanSquaredError()

if __name__ == '__main__':
    unit = 8
    input = 32
    batch_size = 4

    x = np.ones(shape=(batch_size, input))
    y = np.ones(shape=(batch_size, 1))
    # b = np.ones(shape=(input, unit))
    #
    # x = np.dot(a, b)
    # exit()
    l_1 = Dense(input, 8)
    l_2 = Dense(8, 1)

    for i in range(32):
        x_1 = l_1.forward(x)
        x_2 = l_2.forward(x_1)

        error = mse.derivative(x_2, y)
        error = l_2.backward(error, .001)
        l_1.backward(error, .001)

        print(mse.calc(x_2, y))
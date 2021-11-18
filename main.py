import matplotlib.pyplot as plt
import numpy as np
from res.Dense import Dense, ReLu, Sigmoid
from res.loss.MSE import MSE

inputs = np.asarray([
    "Test; This is a Test Message",
    "This is Another Test Message",
    "This One Also",
    "Also This",
])

input = np.asarray([
    "This is Another Input Test",
    "This is Another Input Test One",
])


class Model:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def train(self, x, y, epochs=10, batch_size=1, learning_rate=0.01):
        logit = x
        for layer in self.layers:
            logit = layer.forward(logit)
        loss = MSE()
        grad = loss.gradient(logit, y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        print(f"loss :{np.mean(loss(logit, y))}")
        print(logit)
        return np.mean(loss(logit, y))


if __name__ == '__main__':
    model = Model([
        Dense(2, 8),
        ReLu(),
        Dense(8, 1),
        Sigmoid()
    ])

    test_x = np.random.random(size=(1024, 2))
    test_y = np.asarray([[x[0] == x[1]] for x in test_x], dtype=int)


    history = []
    for x in range(32):
        loss = model.train(test_x, test_y)
        history.append(loss)
    plt.plot(history)
    plt.show()

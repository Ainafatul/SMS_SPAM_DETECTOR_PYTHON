import matplotlib.pyplot as plt

from res.Dense import Dense, Linear, ReLu
from res.dataset.DatasetGenerator import DatasetGenerator
from res.loss.MSE import MSE
from res.models.Model import Model

if __name__ == '__main__':
    model = Model([
        Dense(2, 8),
        ReLu(),
        Dense(8, 1),
        ReLu(),
        # Linear()
        # Sigmoid()
    ])

    model.compile(MSE(), learning_rate=1e-3)

    generator = DatasetGenerator()
    x, y = generator(10240)

    history = model.train(x, y, epochs=16, batch_size=1024)
    plt.plot(history)
    plt.show()

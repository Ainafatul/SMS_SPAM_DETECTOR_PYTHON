import numpy as np

from res.layers.Layer import Layer


class EmbeddingLayer(Layer):

    def forward(self, x):
        return self.__call__(x)

    def __init__(self, input_size, output_size, length):
        self.weight = np.random.uniform(low=-1, high=1, size=output_size)
        self.vocab_size = input_size - 1
        self.size = length

    def add_padding(self, input):
        return np.pad(input, (0, self.size - input.shape[0]), 'constant', constant_values=0)

    def normalize(self, x):
        mean = self.vocab_size / 2
        return (x - mean) / mean

    def __call__(self, x: np.array, *args, **kwargs):
        results = []
        for e in x:
            e = np.asarray(e)
            if e.shape[0] < self.size:
                results.append(self.add_padding(e))
            else:
                results.append(e[:self.size])
        return self.normalize(np.asarray(results))

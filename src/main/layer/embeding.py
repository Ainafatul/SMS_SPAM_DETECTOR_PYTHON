import numpy as np


class Embedding:

    def __init__(self, vocab_size, embedding_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = np.random.randn(self.embedding_size)

    def forward(self, x, training):
        return ((x * self.embedding) / self.embedding_size) - .5

    def backward(self, dout, learning_rate):
        return dout

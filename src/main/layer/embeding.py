import numpy as np


class Embedding:

    def __init__(self, vocab_size, embedding_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = np.random.randn(self.embedding_size)

    def forward(self, x):
        return x * self.embedding

    def backward(self, dout, learning_rate):
        return dout


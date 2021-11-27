import numpy as np


class Embedding:

    def __init__(self, vocab_size, embedding_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = np.random.randn(self.embedding_size)

    # lookup tabel dengan mengkalikan x(panjang seq)
    def forward(self, x, training):
        # dilakukan normalisasi, dg pembagian untuk setiap kata, minus 0,5 agar geser ke 0 kemudian dikali 2
        return (((x * self.embedding) / self.embedding_size) - .5) * 2

    def backward(self, dout, learning_rate):
        return dout

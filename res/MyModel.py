from res import TextEncoder
from res.layers.EmbeddingLayer import EmbeddingLayer


class MyModel:

    def __init__(self, encoder: TextEncoder):
        self.encoder = encoder
        self.embedding = EmbeddingLayer(len(encoder.dictionary), 64, length=8)

    def __call__(self, x):
        x = self.encoder(x)
        x = self.embedding(x)
        return x


import numpy as np
import re

from res.layers.Layer import Layer


def preprocess_sentences(sentences):
    return [re.sub(r'[^a-zA-Z0-9\s]', '', sentence).lower().split() for sentence in sentences]


class TextEncoder(Layer):

    def forward(self, x):
        return self.__call__(x)

    dictionary = None

    def adapt(self, dataset: np.array):
        docs = preprocess_sentences(dataset)
        ledger = {}
        for doc in docs:
            for word in doc:
                if word not in ledger:
                    ledger[word] = 1
                else:
                    ledger[word] += 1
        items = sorted(ledger.items(), key=lambda x: x[1], reverse=True)
        items.insert(0, ("<UNK>", 0))
        self.dictionary = np.asarray([i[0] for i in items])

    def __init__(self, dataset: np.array):
        self.adapt(dataset)

    def __call__(self, x, *args, **kwargs):
        sentences = preprocess_sentences(x)
        results = []
        for sentence in sentences:
            result = []
            for word in sentence:
                if word in self.dictionary:
                    result.append(np.where(self.dictionary == word)[0][0])
                else:
                    result.append(0)
            results.append(result)
        return results

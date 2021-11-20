import re

import numpy as np


class TextVectorization:
    def __init__(self, vocab_size, max_length):
        self.vocab = {}
        self.vocab_size = vocab_size
        self.max_length = max_length

    # function to create a dictionary of words and their indices sorted by top frequency
    # limit dict size to self.vocab_size
    # where the 0th index is reserved for the unknown word
    def fit(self, sentences):
        self.vocab['<unk>'] = 0
        self.vocab_count = {'<unk>': 0}
        for sentence in sentences:
            text = self.filter(sentence)
            for word in text.split():
                if word in self.vocab_count:
                    self.vocab_count[word] += 1
                else:
                    self.vocab_count[word] = 1
        self.vocab_count = sorted(self.vocab_count.items(), key=lambda x: x[1], reverse=True)
        for i, (word, count) in enumerate(self.vocab_count):
            if i >= self.vocab_size:
                break
            self.vocab[word] = i + 1
        return self

    def filter(self, sentence):
        regex = re.compile('[^a-z ]')
        return re.sub(regex, '', sentence.lower())

    # function to convert a sentence to a vector of int from the vocabulary index
    def transform(self, sentence):
        text = self.filter(sentence)
        vector = np.zeros(self.max_length, dtype=np.int32)
        for i, word in enumerate(text.split()):
            if i >= self.max_length:
                break
            if word in self.vocab:
                vector[i] = self.vocab[word]
            else:
                vector[i] = 0
        return vector

    # function to convert sentences to vectors of int from the vocabulary index
    # check if x is a np.array or a string
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return np.array([self.transform(sentence) for sentence in x])
        else:
            return np.asarray(self.transform(x))

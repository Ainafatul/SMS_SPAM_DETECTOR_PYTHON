import abc
from abc import ABC


class Layer(ABC):

    @property
    @abc.abstractmethod
    def input_shape(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_shape(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def input(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError

    def compile(self,input_shape):
        raise NotImplementedError

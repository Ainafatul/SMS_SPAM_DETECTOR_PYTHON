import abc
import time
from abc import ABC

from utils import default_return


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
        return x

    def backward(self, output_error, learning_rate):
        return default_return(output_error)

    def compile(self, input_shape):
        raise NotImplementedError

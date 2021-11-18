from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)

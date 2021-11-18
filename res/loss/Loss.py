from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):

    @abstractmethod
    def __call__(self, y_pred, y_true):
        pass

    @abstractmethod
    def gradient(self, y_pred, y_true):
        pass

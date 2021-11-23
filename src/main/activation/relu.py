from activation import Activation


class ReLU(Activation):

    def forward_activation(self, x):
        return max(0, x)

    def backward_activation(self, x):
        return max(0, x)

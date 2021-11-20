from res.layer.Layer import Layer


class Activation(Layer):

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

    def update(self, learning_rate):
        pass

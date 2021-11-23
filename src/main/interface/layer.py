class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError

from layer import Layer


class Activation(Layer):
    input = None
    input_shape = None
    output_shape = None

    def __call__(self, x, derivative=False):
        if derivative:
            return self.backward_activation(x)
        return self.forward_activation(x)

    def forward(self, input_data):
        self.input = input_data
        self.output = self.forward_activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        return self.backward_activation(self.input) * output_error

    def forward_activation(self, x):
        raise NotImplementedError

    def backward_activation(self, x):
        raise NotImplementedError

    def compile(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape

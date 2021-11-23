from layer import Layer


class Activation(Layer):

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

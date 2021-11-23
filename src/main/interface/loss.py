class Loss:

    def __call__(self, y_pred, y_true, derivative=False):
        if derivative:
            return self.derivative(y_pred, y_true)
        return self.calc(y_pred, y_true)

    def calc(self, y_pred, y_true):
        raise NotImplementedError

    def derivative(self, y_pred, y_true):
        raise NotImplementedError

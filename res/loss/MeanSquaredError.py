class MeanSquaredError:
    def forward(self, y_pred, y_true):
        return ((y_pred - y_true) ** 2).mean()

    def prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

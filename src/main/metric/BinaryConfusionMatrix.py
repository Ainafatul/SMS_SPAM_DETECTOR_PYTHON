import numpy as np


class BinaryConfusionMatrix:

    # penentuan label berdasarkan treshold, diatas 0.5 spam, dibawah 0.5 non spam
    def __init__(self, threshold=0.5):
        # inisialisasi matrix dengan nilai 0 dan ukuran 2 dg 2
        self.matrix = np.zeros((2, 2))
        self.threshold = threshold

    def update(self, y_pred, y_true):
        # setiap y_p didalam ypred, setiap yt didalam ytrue
        for y_p, y_t in zip(y_pred, y_true):
            # yp dirubah menjadi int (awalnya float)lebih besar dari treshold yang ditentukan kemudian dibandingkan dg yt
            # jika spam 1, bukan spam 0
            self.matrix[int(y_p > self.threshold)][y_t] += 1

    def get_accuracy(self):
        # jumlah daari diagonal matrik dibagi dengan jumlah matrik binary
        return np.sum(np.diag(self.matrix)) / np.sum(self.matrix)

    def reset(self):
        # mengembalikan nilai matrix dengan 0
        self.matrix = np.zeros((2, 2))


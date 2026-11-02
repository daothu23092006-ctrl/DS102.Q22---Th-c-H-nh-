import numpy as np
from tqdm import tqdm

class SVM:
    def __init__(self, C: float, epochs: int = 1000, lr: float = 0.05):
        self.C = C
        self.epochs = epochs
        self.lr = lr

        self.W = None
        self.b = 0
                
        self.losses = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        _, dim = X.shape
        self.W = np.random.randn(dim, 1) * 0.05
        # self.b = 0

        y = y.reshape(-1, 1)

        pbar = tqdm(range(self.epochs), desc="Training")
        for _ in pbar:
            idx = np.random.permutation(len(X))
            X = X[idx]
            y = y[idx]
            for ith, x_i in enumerate(X):
                x_i = x_i.reshape(-1, 1) 

                condition = (y[ith] * (self.W.T @ x_i + self.b)).item()

                if condition >= 1:
                    dw = self.W
                    db = 0
                else:
                    dw = self.W - self.C * y[ith] * x_i
                    db = -self.C * y[ith]

                self.W -= self.lr * dw
                self.b -= self.lr * db

            y_hat = self.predict(X)
            loss = self.hinge_loss(y, y_hat)
            self.losses.append(loss)

            pbar.set_postfix({
                "loss": loss
            })

    def predict(self, X: np.ndarray):
        y_hat = (self.W.T @ X.T + self.b).T

        return y_hat

    def hinge_loss(self, y: np.ndarray, y_hat: np.ndarray):  
        delta = 1 - y * y_hat
        hinge = np.where(delta > 0, delta, 0)

        return 0.5 * np.sum(self.W ** 2) + self.C * hinge.mean()
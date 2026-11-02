import numpy as np
from tqdm import tqdm

class SoftmaxRegression:
    def __init__(self, epoch:int, lr:float):
        self.epoch = epoch
        self.lr = lr
        self.w = None
        self.losses = []

    def loss_fn(self, y:np.ndarray,y_hat:np.ndarray)->float:
        # y,y_hat:(N,K)
        return -(y*np.log(y_hat + 1e-15)).sum(axis=1).mean()
    
    def fit(self,X:np.ndarray,y:np.ndarray):
        _,d = X.shape
        _,k = y.shape
        self.w =np.zeros((d,k),dtype=np.float64)
        e = 0
        with tqdm(range(self.epoch)) as pbar:
            for _ in pbar: 
                #forward pass
                y_hat = self.predict(X)  

                #backward pass
                delta_y = (y_hat - y)         #(N,k)
                N = X.shape[0]
                gradient = (X.T @ delta_y) / N
                self.w = self.w - self.lr * gradient

                l = self.loss_fn(y,y_hat)
                self.losses.append(l)

                pbar.set_postfix({
                    "loss": l
                })
                e += 1

    def softmax(self,z:np.ndarray):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def predict(self,X:np.ndarray)->np.ndarray:
        z = X @ self.w                       #(N,1)
        y_hat = self.softmax(z)
        return y_hat
    
    def predict_label(self, X):
        y_hat = self.predict(X)
        return np.argmax(y_hat, axis=1)
    
    def evaluate(self, y, y_hat) -> dict:
        y_pred = np.argmax(y_hat, axis=1)

        precision_list = []
        recall_list = []
        f1_list = []

        num_classes = len(np.unique(y))

        for c in range(num_classes):
            TP = np.sum((y == c) & (y_pred == c))
            FP = np.sum((y != c) & (y_pred == c))
            FN = np.sum((y == c) & (y_pred != c))

            precision = TP / (TP + FP + 1e-15)
            recall = TP / (TP + FN + 1e-15)
            f1 = 2 * precision * recall / (precision + recall + 1e-15)

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        return {
            "precision": np.mean(precision_list),
            "recall": np.mean(recall_list),
            "f1": np.mean(f1_list),
        }

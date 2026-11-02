import numpy as np
from tqdm import tqdm

class LogisticRegression:
    def loss_fn(self, y:np.ndarray,y_hat:np.ndarray)->float:
        l =(1-y)*np.log(1-y_hat + 1e-15) + y*np.log(y_hat + 1e-15)
        return -l.mean()
       
    def __init__(self, epoch:int, lr:float):
      self.epoch = epoch
      self.lr = lr
      self.losses = []
      self._w = None
      pass

  
    def fit(self,X:np.ndarray,y:np.ndarray):
      N,d = X.shape
      self._w = np.zeros((d,1),dtype=np.float64)
      e = 0
      y = y.reshape(-1,1)
      with tqdm(range(self.epoch), desc=f"Epoch {e+1}:") as pbar:
        for e in pbar: 
            #forward pass
            y_hat = self.predict(X)

            #backward pass
            delta_y = (y_hat - y)           #(N,1)
            gradient = (X.T @ delta_y) / N  #(1,d)
            self._w = self._w - self.lr * gradient  #(d,1)

            l = self.loss_fn(y,y_hat)
            self.losses.append(l)

            pbar.set_description(f"Epoch {e+1}")
            pbar.set_postfix({
            "loss": l
            })

    def sigmoid(self,z:np.ndarray)->np.ndarray:
      return 1/(1+np.exp(-z))

    def predict(self,X:np.ndarray)->np.ndarray:
      z = X @ self._w                       #(N,1)
      y_hat = self.sigmoid(z)
      return y_hat

    def evaluate(self, y, y_hat) -> dict:
      y = y.reshape(-1,1)
      y_pred = (y_hat >= 0.5).astype(int)

      TP = np.sum((y == 1) & (y_pred == 1))
      FP = np.sum((y == 0) & (y_pred == 1))
      FN = np.sum((y == 1) & (y_pred == 0))

      precision = TP / (TP + FP + 1e-15)
      recall = TP / (TP + FN + 1e-15)
      f1 = 2 * precision * recall / (precision + recall + 1e-15)

      return {
          "precision": precision,
          "recall": recall,
          "f1": f1,
      }


# Khi phân loại nhiều nhãn thì dùng micro avg => bộ dữ liệu mất cân bằng thì dự đoán sai nên phải dùng micro avg
# Train loss phải giảm,nếu không giảm thì không học được gì hết
# train ra độ chính xác 100% trong bộ dữ liệu lớn => sai 
import os
path = 'C:/Users/admin/Downloads/project/data'
print("Files in data folder:")
print(os.listdir(path))

import numpy as np
import idx2numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

train_images = idx2numpy.convert_from_file(
    'C:/Users/admin/Downloads/project/data/train-images.idx3-ubyte'
)
train_labels = idx2numpy.convert_from_file(
    'C:/Users/admin/Downloads/project/data/train-labels.idx1-ubyte'
)

test_images = idx2numpy.convert_from_file(
    'C:/Users/admin/Downloads/project/data/t10k-images.idx3-ubyte'
)
test_labels = idx2numpy.convert_from_file(
    'C:/Users/admin/Downloads/project/data/t10k-labels.idx1-ubyte'
)

mask_train = (train_labels == 0) | (train_labels == 1)
mask_test = (test_labels == 0) | (test_labels == 1)

X_train = train_images[mask_train]
y_train = train_labels[mask_train]

X_test = test_images[mask_test]
y_test = test_labels[mask_test]

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

X_train = X_train / 255.0
X_test = X_test / 255.0

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=== Logistic Regression (0 vs 1) ===")
print(f"precision: {precision:.4f}")
print(f"recall: {recall:.4f}")
print(f"f1: {f1:.4f}")
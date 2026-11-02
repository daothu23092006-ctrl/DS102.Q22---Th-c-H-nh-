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

X_train = train_images.reshape(train_images.shape[0], -1)
X_test = test_images.reshape(test_images.shape[0], -1)

X_train = X_train / 255.0
X_test = X_test / 255.0

model = LogisticRegression(max_iter=1000)

model.fit(X_train, train_labels)

y_pred = model.predict(X_test)

precision = precision_score(test_labels, y_pred, average='macro')
recall = recall_score(test_labels, y_pred, average='macro')
f1 = f1_score(test_labels, y_pred, average='macro')

print("=== Softmax Regression (10 classes) ===")
print(f"precision: {precision:.4f}")
print(f"recall: {recall:.4f}")
print(f"f1: {f1:.4f}")
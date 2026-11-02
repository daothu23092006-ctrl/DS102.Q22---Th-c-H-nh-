import os
path = 'C:/Users/admin/Downloads/project/data'
print("Files in data folder:")
print(os.listdir(path))

import numpy as np
import idx2numpy
from softmax_regression import SoftmaxRegression
from matplotlib import pyplot as plt
import seaborn as sns

def convert_to_onehot_vector(labels: np.ndarray):
    N = labels.shape[0]
    print(labels.shape)
    total_classes = labels.max() + 1
    oh_labels = np.zeros((N, total_classes))
    oh_labels[np.arange(N), labels] = 1
    
    return oh_labels

#loading data
train_images = idx2numpy.convert_from_file(
    'C:/Users/admin/Downloads/project/data/train-images.idx3-ubyte'
)
train_labels = idx2numpy.convert_from_file(
    'C:/Users/admin/Downloads/project/data/train-labels.idx1-ubyte'
)    
encoded_train_labels = convert_to_onehot_vector(train_labels)
print((train_labels[10]))
print((encoded_train_labels[10]))

train_data = ((train_images, train_labels))

test_images = idx2numpy.convert_from_file(
    'C:/Users/admin/Downloads/project/data/t10k-images.idx3-ubyte'
)
test_labels = idx2numpy.convert_from_file(
    'C:/Users/admin/Downloads/project/data/t10k-labels.idx1-ubyte'
)
test_data = ((test_images, test_labels))

#flatten images
N,_, _ = train_images.shape
train_images = train_images.reshape(N, -1)
N,_, _ = test_images.shape
test_images = test_images.reshape(N, -1)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = SoftmaxRegression(
    epoch=500,
    lr = 0.01
)

model.fit(train_images, encoded_train_labels)
y_hat = model.predict(test_images)
metrics = model.evaluate(test_labels, y_hat)

for k, v in metrics.items():
    print(f"{k}: {v}")

plt.plot(model.losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
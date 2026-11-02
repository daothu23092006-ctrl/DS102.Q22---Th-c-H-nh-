import os
path = 'C:/Users/admin/Downloads/project/data'
print("Files in data folder:")
print(os.listdir(path))

import numpy as np
import idx2numpy
from logistic_regression import LogisticRegression

train_images = idx2numpy.convert_from_file(
    'C:/Users/admin/Downloads/project/data/train-images.idx3-ubyte'
)
train_labels = idx2numpy.convert_from_file(
    'C:/Users/admin/Downloads/project/data/train-labels.idx1-ubyte'
)    
train_data = (train_images, train_labels)

test_images = idx2numpy.convert_from_file(
    'C:/Users/admin/Downloads/project/data/t10k-images.idx3-ubyte'
)
test_labels = idx2numpy.convert_from_file(
    'C:/Users/admin/Downloads/project/data/t10k-labels.idx1-ubyte'
)
test_data = (test_images, test_labels)

def filter_data(data, condition):
    images, labels = data

    new_images = images[labels == condition]
    new_labels = labels[labels == condition]

    return new_images, new_labels

train_images_0, train_labels_0 = filter_data(train_data, 0)
train_images_1, train_labels_1 = filter_data(train_data, 1) 
train_images = np.concatenate(
    (train_images_0, train_images_1), 
    axis=0
    )

test_images_0, test_labels_0 = filter_data(test_data, 0)
test_images_1, test_labels_1 = filter_data(test_data, 1)
test_images = np.concatenate(
    (test_images_0, test_images_1), 
    axis=0
    )

train_labels = np.concatenate(
    (train_labels_0, train_labels_1),       
    axis=0
)

test_labels = np.concatenate(
    (test_labels_0, test_labels_1),
    axis=0
)

# flatten the images
N,_, _ = train_images.shape
train_images = train_images.reshape(N, -1)
N,_, _ = test_images.shape
test_images = test_images.reshape(N, -1)

model = LogisticRegression(
    epoch = 500,
    lr =  0.1
)

train_images = train_images / 255.0
test_images = test_images / 255.0

perm = np.random.permutation(len(train_images))
train_images = train_images[perm]
train_labels = train_labels[perm]

model.fit(train_images, train_labels)
predicted_labels = model.predict(test_images)
metrics = model.evaluate(test_labels, predicted_labels)
for metric in metrics:
    print(f"{metric}: {metrics[metric]}")






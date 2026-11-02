import cv2 as cv
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from svm import SVM

BASE_DIR = "c:/Users/admin/Downloads/projectlab3/data/chest_xray"

def load_data(split: str = "train"):
    normal = "NORMAL"
    pneumonia = "PNEUMONIA"

    images = []
    labels = []

    for img_file in os.listdir(os.path.join(BASE_DIR, split, normal)):
        image = cv.imread(os.path.join(BASE_DIR, split, normal, img_file))
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, (128, 128), interpolation=cv.INTER_AREA)
        image = image / 255.0
        images.append(image.flatten())
        labels.append(1)

    for img_file in os.listdir(os.path.join(BASE_DIR, split, pneumonia)):
        image = cv.imread(os.path.join(BASE_DIR, split, pneumonia, img_file))
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, (128, 128), interpolation=cv.INTER_AREA)
        image = image / 255.0
        images.append(image.flatten())
        labels.append(-1)

    images = np.stack(images, axis=0)
    return {
        "images": images,
        "labels": np.expand_dims(np.array(labels), axis=-1)
    }

print("Start loading train...")
train_data = load_data(split="train")

print("Start loading val...")
dev_data = load_data(split="val")

print("Start loading test...")
test_data = load_data(split="test")

print("Start training...")

model = SVM(
    C = 1,
    epochs = 20,
    lr = 1e-3
)

model.fit(train_data["images"], train_data["labels"])
y_hat = model.predict(test_data["images"])
y_hat = np.where(y_hat < 0, -1, 1)

P = precision_score(test_data["labels"].ravel(), y_hat.ravel())
R = recall_score(test_data["labels"].ravel(), y_hat.ravel())
f1 = f1_score(test_data["labels"].ravel(), y_hat.ravel())

print(f"Precision: {P}")
print(f"Recall: {R}")
print(f"F1: {f1}")

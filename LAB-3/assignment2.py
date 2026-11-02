import cv2 as cv
import os
import numpy as np
from numpy.random import normal
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

BASE_DIR = "c:/Users/admin/Downloads/projectlab3/data/chest_xray"

def load_data(split: str = "train"):
    normal = "NORMAL"
    pneumonia = "PNEUMONIA"

    images = []
    labels = []

    for img_file in os.listdir(os.path.join(BASE_DIR, split, normal)):
        image = cv.imread(os.path.join(BASE_DIR, split, normal, img_file))
        if image is None:
            continue
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, (128, 128))
        image = image / 255.0
        images.append(image.flatten())
        labels.append(1)

    for img_file in os.listdir(os.path.join(BASE_DIR, split, pneumonia)):
        image = cv.imread(os.path.join(BASE_DIR, split, pneumonia, img_file))
        if image is None:
            continue
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, (128, 128))
        image = image / 255.0
        images.append(image.flatten())
        labels.append(-1)

    images = np.stack(images, axis=0)
    return {
        "images": images,
        "labels": np.array(labels)
    }

print("Loading data...")
train_data = load_data("train")
test_data = load_data("test")

print("Training sklearn SVM...")

scaler = StandardScaler()

X_train = scaler.fit_transform(train_data["images"])
X_test = scaler.transform(test_data["images"])

model = LinearSVC(C=1, max_iter=1000)
model.fit(X_train, train_data["labels"])

y_hat = model.predict(X_test)

P = precision_score(test_data["labels"], y_hat)
R = recall_score(test_data["labels"], y_hat)
f1 = f1_score(test_data["labels"], y_hat)

P1, R1, f1_1 = 0, 0, 0
print(f"Precision: {P}")
print(f"Recall: {R}")
print(f"F1: {f1}")

print("\n ****** So sánh ******")
print(f"{'Metric':<12} {'Assignment 1':>14} {'Assignment 2':>14}")
print("-" * 42)
print(f"{'Precision':<12} {P1:>14.4f} {P:>14.4f}")
print(f"{'Recall':<12} {R1:>14.4f} {R:>14.4f}")
print(f"{'F1':<12} {f1_1:>14.4f} {f1:>14.4f}")

print("\n ****** Điều kiện so sánh ******")
print(f"  C       = 1")
print(f"  epochs  = 1000")
print(f"  lr      = 1e-4")

print("\n ******Nhận xét ******")

# Precision
if P1 > P:
    print(f"  - Precision: Assignment 1 cao hơn ({P1:.4f} > {P:.4f})")
elif P1 < P:
    print(f"  - Precision: Assignment 2 cao hơn ({P:.4f} > {P1:.4f})")
else:
    print(f"  - Precision: Hai model bằng nhau ({P:.4f})")

# Recall
if R1 > R:
    print(f"  - Recall: Assignment 1 cao hơn ({R1:.4f} > {R:.4f})")
elif R1 < R:
    print(f"  - Recall: Assignment 2 cao hơn ({R:.4f} > {R1:.4f})")
else:
    print(f"  - Recall: Hai model bằng nhau ({R:.4f})")

# F1
if f1_1 > f1:
    print(f"  - F1: Assignment 1 cao hơn ({f1_1:.4f} > {f1:.4f})")
elif f1_1 < f1:
    print(f"  - F1: Assignment 2 cao hơn ({f1:.4f} > {f1_1:.4f})")
else:
    print(f"  - F1: Hai model bằng nhau ({f1:.4f})")
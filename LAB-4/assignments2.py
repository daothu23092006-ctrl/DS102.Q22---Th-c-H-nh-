import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from random_forest import RandomForestClassifier

red   = pd.read_csv('wine+quality/winequality-red.csv', sep=';')
white = pd.read_csv('wine+quality/winequality-white.csv', sep=';')

df = pd.concat([red, white], ignore_index=True)

X = df.drop(columns='quality').values.astype(np.float64)
y = df['quality'].values

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

classes = np.unique(y)
label_map = {c: i for i, c in enumerate(classes)}
y_encoded = np.array([label_map[v] for v in y])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Train
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=3,
    max_features="sqrt"
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)

f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n=== Random Forest Evaluation ===")
print(f"F1-Score (Macro): {f1_macro:.4f}")
print(f"F1-Score (Weighted): {f1_weighted:.4f}")
print("\nDetailed report:")
print(classification_report(y_test, y_pred, zero_division=0))

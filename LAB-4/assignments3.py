import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

red   = pd.read_csv('wine+quality/winequality-red.csv',   sep=';')
white = pd.read_csv('wine+quality/winequality-white.csv', sep=';')

df = pd.concat([red, white], ignore_index=True)

X = df.drop(columns='quality').values.astype(np.float64)
y = df['quality'].values

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Decision Tree 
dt = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=3,
    random_state=42
)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Random Forest 
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluation
print("\n Decision Tree (sklearn) evaluation")
print(f"F1-Score (Macro): {f1_score(y_test, y_pred_dt, average='macro',    zero_division=0):.4f}")
print(f"F1-Score (Weighted): {f1_score(y_test, y_pred_dt, average='weighted', zero_division=0):.4f}")
print(classification_report(y_test, y_pred_dt, zero_division=0))

print("\n Random Forest (sklearn) evaluation")
print(f"F1-Score (Macro): {f1_score(y_test, y_pred_rf, average='macro',    zero_division=0):.4f}")
print(f"F1-Score (Weighted): {f1_score(y_test, y_pred_rf, average='weighted', zero_division=0):.4f}")
print(classification_report(y_test, y_pred_rf, zero_division=0))
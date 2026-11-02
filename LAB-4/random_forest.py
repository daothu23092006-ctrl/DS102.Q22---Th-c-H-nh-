import numpy as np
from decision_tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(
            self,
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt"
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def _get_max_features(self, n_features):
        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        return n_features

    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]
        max_feat = self._get_max_features(n_features)

        for _ in range(self.n_estimators):
            feat_indices = np.random.choice(n_features, size=max_feat, replace=False)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample[:, feat_indices], y_sample)
            self.trees.append((tree, feat_indices))

    def predict(self, X):
        tree_preds = np.array([
            tree.predict(X[:, feat_indices])
            for tree, feat_indices in self.trees
        ])

        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=tree_preds
        )
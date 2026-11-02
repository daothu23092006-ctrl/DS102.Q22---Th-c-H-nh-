import numpy as np

class KMeans:
    def __init__(self):
        self.centroids = []

    def _init_centroids(self, X, k):
        # Randomly pick k rows of X ans initial centroids
        return X[np.random.choice(X.shape[0], k, replace=False)]
    
    def _assign_labels(self, X, centroids):
        D = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(D, axis=1)
    
    def _update_centroids(self, X, labels, k):
        centroids = np.zeros((k, X.shape[1]))
        for i in range(k):
            # Collect all points assigned to the k-th cluster
            Xk = X[labels == i]
            # Take average
            centroids[i] = np.mean(Xk, axis=0)
        return centroids
    
    def has_converged(self, centroids, new_centroids):
        # Return True if two sets of centroids are the same
        return set([tuple(a) for a in centroids]) == set([tuple(a) for a in new_centroids])
    
    def fit(self, X, k):
        current_centroids = self._init_centroids(X, k)
        self.centroids.append(current_centroids)
        labels = []
        it = 0
        while True:
            labels.append(self._assign_labels(X, current_centroids))
            new_centroids = self._update_centroids(X, labels[-1], k)
            if self.has_converged(current_centroids, new_centroids):
                break
            self.centroids.append(new_centroids) 
            it += 1 
            current_centroids = new_centroids
        return labels, it
    
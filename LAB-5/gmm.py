# Gaussian Mixture Model
import numpy as np

class GMM:
    def __init__(self, n_components, max_iters=100, comp_names=None):
        self.n_componets = n_components         
        self.max_iter = max_iters               

        if comp_names is None:                  
            self.comp_names = [f"comp({index})" for index in range(self.n_componets)]
        else:
            self.comp_names = comp_names

        self.pi = [1 / self.n_componets for _ in range(self.n_componets)]

    def multivariate_normal(self, X, mean_vector, covariance_matrix):
        d = len(X)
        diff = X - mean_vector
        norm_const = 1.0 / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance_matrix)))
        exponent = np.exp(-0.5 * diff @ np.linalg.inv(covariance_matrix) @ diff)
        return norm_const * exponent

    def multivariate_normal_vectorized(self, X, mean_vector, covariance_matrix):
        d = X.shape[1]
        diff = X - mean_vector
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance_matrix))
        inv_cov = np.linalg.inv(covariance_matrix)
        exponent = np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=1))
        return norm_const * exponent

    def fit(self, X):
        new_X = np.array_split(X, self.n_componets)
        self.mean_vector = [np.mean(x, axis=0) for x in new_X]
        self.covariance_matrixes = [np.cov(x.T, ddof=0) for x in new_X]  

        for _ in range(self.max_iter):
            # E-step
            self.r = np.zeros((len(X), self.n_componets))
            for n in range(len(X)):
                for k in range(self.n_componets):
                    self.r[n][k] = self.pi[k] * self.multivariate_normal(
                        X[n], self.mean_vector[k], self.covariance_matrixes[k]
                    )
                self.r[n] /= np.sum(self.r[n])
            
            # M-step
            N = np.sum(self.r, axis=0)
            self.mean_vector = np.zeros((self.n_componets, X.shape[1]))
            for k in range(self.n_componets):
                for n in range(len(X)):
                    self.mean_vector[k] += self.r[n][k] * X[n]
                self.mean_vector[k] /= N[k]   

            self.covariance_matrixes = [np.zeros((X.shape[1], X.shape[1])) for _ in range(self.n_componets)]
            for k in range(self.n_componets):
                self.covariance_matrixes[k] = np.cov(X.T, aweights=self.r[:, k], ddof=0)
            self.pi = [N[k] / len(X) for k in range(self.n_componets)]

    def fit_fast(self, X):
        new_X = np.array_split(X, self.n_componets)
        self.mean_vector = np.array([np.mean(x, axis=0) for x in new_X])
        self.covariance_matrixes = [np.cov(x.T, ddof=0) for x in new_X]

        for _ in range(self.max_iter):
            # E-step (vectorized)
            self.r = np.zeros((len(X), self.n_componets))
            for k in range(self.n_componets):
                self.r[:, k] = self.pi[k] * self.multivariate_normal_vectorized(
                    X, self.mean_vector[k], self.covariance_matrixes[k]
                )            # Normalize r (mỗi hàng chia cho tổng)
            self.r /= self.r.sum(axis=1, keepdims=True)

            # M-step
            N = np.sum(self.r, axis=0)

            # Cập nhật mean_vector (vectorized)
            self.mean_vector = (self.r.T @ X) / N[:, np.newaxis]

            # Cập nhật covariance_matrixes
            self.covariance_matrixes = []
            for k in range(self.n_componets):
                self.covariance_matrixes.append(
                    np.cov(X.T, aweights=self.r[:, k], ddof=0)
                )

            self.pi = [N[k] / len(X) for k in range(self.n_componets)]

    def predict(self, X):
        probas = []
        for n in range(len(X)):
            probas.append(
                [self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])
                 for k in range(self.n_componets)]
            )
        cluster = []
        for proba in probas:
            cluster.append(self.comp_names[proba.index(max(proba))])
        return cluster
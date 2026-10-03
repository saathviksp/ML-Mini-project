import numpy as np

class GaussianDiscriminantAnalysis:
    """
    Gaussian Discriminant Analysis (GDA) classifier implemented from scratch.
    
    Model parameters:
    - phi: P(y=1) prior probability of positive class
    - mu0: Mean vector for class 0 (non-pulsar), shape (d,)
    - mu1: Mean vector for class 1 (pulsar), shape (d,)
    - sigma: Shared covariance matrix, shape (d, d)
    """
    def __init__(self):
        self.phi = None
        self.mu0 = None
        self.mu1 = None
        self.sigma = None
        self.sigma_inv = None
        self.log_det_sigma = None

    def fit(self, X, y):
        """
        Fits GDA parameters phi, mu0, mu1, and shared sigma.
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        n1 = np.sum(y == 1)
        n0 = np.sum(y == 0)

        # 1. Class prior phi
        self.phi = n1 / n

        # 2. Mean vectors mu0 and mu1
        self.mu0 = np.sum(X[y == 0], axis=0) / n0
        self.mu1 = np.sum(X[y == 1], axis=0) / n1

        # 3. Shared Covariance Matrix Sigma
        X_centered = np.zeros_like(X)
        X_centered[y == 0] = X[y == 0] - self.mu0
        X_centered[y == 1] = X[y == 1] - self.mu1

        self.sigma = (X_centered.T @ X_centered) / n
        
        # Pseudo-inverse and log determinant for numerical stability
        self.sigma_inv = np.linalg.pinv(self.sigma)
        sign, self.log_det_sigma = np.linalg.slogdet(self.sigma)

        return self

    def predict_proba(self, X):
        """
        Returns predicted probability P(y=1|X) for each sample.
        X: numpy array of shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=np.float64)
        
        # Compute Mahalanobis distances
        dev0 = X - self.mu0
        dev1 = X - self.mu1

        # Q_k(x) = log P(x|y=k) + log P(y=k)
        # Quad form: -0.5 * (x - mu_k)^T * Sigma^{-1} * (x - mu_k)
        quad0 = -0.5 * np.sum(dev0 @ self.sigma_inv * dev0, axis=1)
        quad1 = -0.5 * np.sum(dev1 @ self.sigma_inv * dev1, axis=1)

        log_prior0 = np.log(1.0 - self.phi + 1e-15)
        log_prior1 = np.log(self.phi + 1e-15)

        q0 = quad0 + log_prior0
        q1 = quad1 + log_prior1

        # Sigmoid of log likelihood ratio: P(y=1|X) = 1 / (1 + exp(-(q1 - q0)))
        diff = q1 - q0
        # Clip diff to prevent overflow
        diff = np.clip(diff, -500, 500)
        proba_1 = 1.0 / (1.0 + np.exp(-diff))
        
        return proba_1

    def predict(self, X, threshold=0.5):
        """Returns binary predictions based on specified threshold."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

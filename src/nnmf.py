import numpy as np
import numpy.linalg as la

from .base import BaseModel


class NNMFModel(BaseModel):
    def __init__(self, n_components, max_iter=1000, epsilon=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.epsilon = epsilon

    def fit(self, X, y=None, *args, **kwargs):
        self.fit_transform(X, y, args=args, kwargs=kwargs)
        return self

    def transform(self, X):
        return self._nnmf_method(X)

    def fit_transform(self, X, y=None, *args, **kwargs):
        return self._nnmf_method(X)

    def _init_random_w_h(self, input_shape, hiden_shape, output_shape):
        W = np.random.randint(100, size=(input_shape, hiden_shape))
        H = np.random.randint(100, size=(hiden_shape, output_shape))
        return W, H

    def _nnmf_method(self, X):
        norms = []
        W, H = self._init_random_w_h(X.shape[0], self.n_components, X.shape[1])

        X_old = W @ H
        X_new = X_old.copy()

        for _ in range(self.max_iter):
            alpha = H / (W.T @ W @ H)
            beta = W / (W @ H @ (H.T))

            H = H - alpha * (W.T @ W @ H - W.T @ X)
            W = W - beta * (W @ H @ (H.T) - X @ (H.T))

            X_new = W @ H

            norms.append(la.norm(X - X_new))

            if la.norm(X_old - X_new) < self.epsilon:
                break

            X_old = X_new

        self.W = W
        self.H = H
        self.norms = norms

        return self.W

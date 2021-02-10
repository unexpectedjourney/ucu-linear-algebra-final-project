import numpy as np
import numpy.linalg as la
import pandas as pd

from .base import BaseModel
from .data import generate_sparce_matrix


class NNMFModel(BaseModel):
    def __init__(self,
                 n_components,
                 max_iter=1000,
                 epsilon=1e-6,
                 verbose=False):
        self.n_components = n_components
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.verbose = verbose

    def fit(self, X, y=None, *args, **kwargs):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pd.DataFrame object")

        self.coo_matrix = generate_sparce_matrix(X)
        X = self.coo_matrix.toarray()

        self.fit_transform(X, y, args=args, kwargs=kwargs)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pd.DataFrame object")

        self.coo_matrix = generate_sparce_matrix(X)
        X = self.coo_matrix.toarray()

        return self._nnmf_method(X)

    def fit_transform(self, X, y=None, *args, **kwargs):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pd.DataFrame object")

        self.coo_matrix = generate_sparce_matrix(X)
        X = self.coo_matrix.toarray()

        return self._nnmf_method(X)

    @staticmethod
    def get_info_from_final_results(row, matrix):
        movie_id = row.movie_id.astype(int)
        customer_id = row.customer_id.astype(int)
        return np.round(matrix[movie_id, customer_id])

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pd.DataFrame object")
        matrix = self.W @ self.H
        X["rating"] = X.apply(self.get_info_from_final_results, args=(matrix,), axis=1)
        return X

    @staticmethod
    def _init_random_w_h(input_shape, hiden_shape, output_shape):
        W = np.random.randint(100, size=(input_shape, hiden_shape))
        H = np.random.randint(100, size=(hiden_shape, output_shape))
        return W, H

    def _nnmf_method(self, X):
        norms = []
        W, H = self._init_random_w_h(X.shape[0], self.n_components, X.shape[1])

        X_old = W @ H
        X_new = X_old.copy()

        for iteration in range(self.max_iter):
            alpha = H / (W.T @ W @ H)
            beta = W / (W @ H @ (H.T))

            H = H - alpha * (W.T @ W @ H - W.T @ X)
            W = W - beta * (W @ H @ (H.T) - X @ (H.T))

            X_new = W @ H

            norms.append(la.norm(X - X_new))

            if self.verbose:
                print(f"{iteration+1}. Norm: {norms[-1]}")

            if la.norm(X_old - X_new) < self.epsilon:
                break

            X_old = X_new

        self.W = W
        self.H = H
        self.norms = norms

        return self.W

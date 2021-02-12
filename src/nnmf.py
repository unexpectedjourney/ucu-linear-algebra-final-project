import sys

import numpy as np
import numpy.linalg as la
import pandas as pd
from sklearn.preprocessing import normalize
from tqdm import tqdm

from .base import BaseModel
from .data import generate_sparce_matrix


class NNMFModel(BaseModel):
    def __init__(self,
                 n_components,
                 max_iter=1000,
                 epsilon=1e-6,
                 clip=True,
                 verbose=False):
        self.n_components = n_components
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.verbose = verbose
        self.clip = clip
        self.min_value = None
        self.max_value = None

    def fit(self, X, y=None, *args, **kwargs):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pd.DataFrame object")

        self.fit_transform(X, y, args=args, kwargs=kwargs)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pd.DataFrame object")

        X = generate_sparce_matrix(X)

        return self._nnmf_method(X)

    def fit_transform(self, X, y=None, *args, **kwargs):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pd.DataFrame object")

        X = generate_sparce_matrix(X)

        return self._nnmf_method(X)

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pd.DataFrame object")

        preds = []
        desc = "predict loop"
        X_ = self.W @ self.H
        for _, item in tqdm(X.iterrows(), desc=desc, disable=not self.verbose):
            preds.append(X_[item.customer_id, item.movie_id])

        if self.clip:
            preds = np.clip(preds, self.min_value, self.max_value)

        return preds

    @staticmethod
    def _init_random_w_h(input_shape, hiden_shape, output_shape):
        W = np.random.rand(input_shape, hiden_shape)
        H = np.random.rand(hiden_shape, output_shape)

        W = normalize(W)
        return W, H

    def _nnmf_method(self, X):
        if self.clip:
            self.min_value = X.min()
            self.max_value = X.max()

        norms = []
        W, H = self._init_random_w_h(X.shape[0], self.n_components, X.shape[1])

        X_old = W @ H
        X_new = X_old.copy()
        e = np.finfo(np.float64).eps

        desc = "train loop"
        for iteration in tqdm(range(self.max_iter), desc=desc, disable=not self.verbose):
            H = np.multiply(H, ((W.T @ X) / (W.T @ W @ H + e)))
            W = np.multiply(W, ((X @ H.T) / (W @ H @ H.T + e)))

            W = np.nan_to_num(W)
            H = np.nan_to_num(H)

            W = normalize(W)

            X_new = W @ H

            norms.append(la.norm(X - X_new))

            if la.norm(X - X_new) < self.epsilon:
                break

            X_old = X_new

        self.W = W
        self.H = H
        self.norms = norms

        return self.W

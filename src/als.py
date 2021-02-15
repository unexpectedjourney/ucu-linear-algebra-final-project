import numpy as np
import numpy.linalg as la
import pandas as pd

from tqdm.cli import tqdm 
from scipy.sparse import csr_matrix


class AlsSVD:
    def __init__(
        self,         
        k=50, n_epochs=20, reg=0.02, 
        user_field="customer_id",
        item_field="movie_id",
        target_field="rating",
        random_seed=42,
        verbose=True,
    ):
        self.k = k
        self.n_epochs = n_epochs
        self.reg = reg
        self.user_field = user_field
        self.item_field = item_field
        self.target_field = target_field
        self.random_seed = random_seed
        self.verbose = verbose
            
    
    def fit(self, df):
        self.df = df
        self.target_limits = (
            self.df[self.target_field].min(),
            self.df[self.target_field].max(),
        )
        
        self.r = csr_matrix((
            df[self.target_field].values, 
            (df[self.user_field].values, df[self.item_field].values)
        ))
        
        # TODO intialization with custom distribution parameters
        np.random.seed(self.random_seed)
        self.x = np.random.rand(self.k, self.r.shape[0])
        self.y = np.random.rand(self.k, self.r.shape[1])
        
        desc = "train loop"
        for e in tqdm(range(self.n_epochs), desc=desc, disable=not self.verbose):
            self._run_train_epoch()
    
            
    def _run_train_epoch(self):
        for user in self.df[self.user_field].unique():
            u_select = (self.r[user] != 0).toarray().reshape(-1)
            cur_y = self.y[:, u_select]
            cur_r = self.r[user, u_select].toarray()
            
            first_sum = (cur_y @ cur_y.T) \
                      + (self.reg * np.identity(self.k) * cur_y.shape[1])
            second_sum = (cur_y * cur_r).sum(1).reshape(-1, 1)

            res = la.inv(first_sum) @ second_sum
            res = res.reshape(-1)
            self.x[:, user] = res
            
        for item in self.df[self.item_field].unique():
            i_select = (self.r[:, item] != 0).toarray().reshape(-1)
            cur_x = self.x[:, i_select]
            cur_r = self.r[i_select, item].toarray()
            
            first_sum = (cur_x @ cur_x.T) \
                      + (self.reg * np.identity(self.k) * cur_y.shape[1])
            second_sum = (cur_x * cur_r.T).sum(1).reshape(-1, 1)
            
            res = la.inv(first_sum) @ second_sum
            res = res.reshape(-1)
            self.y[:, item] = res

            
    def predict(self, df):
        preds = []
        desc = "predict loop"
        for _, item in tqdm(df.iterrows(), desc=desc, disable=not self.verbose):
            x_vec = self.x[:, item[self.user_field]].reshape(-1, 1)
            y_vec = self.y[:, item[self.item_field]].reshape(-1, 1)
            preds.append((x_vec.T @ y_vec)[0, 0])
        preds = np.array(preds)
        preds = np.clip(preds, *self.target_limits)
        return preds

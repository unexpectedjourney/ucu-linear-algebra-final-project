import numpy as np
import pandas as pd
import time


class Funk(BaseModel):

    def __init__(self, lr=0.005, reg=0.02, n_epochs=20, n_factors=100, min_delta=0.001):
        self.lr = lr  # learning rate
        self.reg = reg  # regularization coef
        self.n_epochs = n_epochs  # number of SGD iterations
        self.n_factors = n_factors  # number of latent factors
        self.min_delta = min_delta  # min iteration improvement, needed for early stop checking
        self.min_rating = 1
        self.max_rating = 5

    def fit(self, X, X_val):
        start = time.time()

        X = self.preprocess_data(X)
        X_val = self.preprocess_data(X_val, train=False)  # for val data train must be False, so we won't update mappings
        self.init_metrics()  # init empty matrix of metrics
        self.mean = np.mean(X[:, 2])  # the mean value of ratings
        self.sgd(X, X_val)  # perform sgd

        end = time.time()
        print(f'this stuff was running {end - start:.1f} sec')

        return self

    def predict(self, X):
        predictions = []  # return predictions as list
        for u_id, i_id in zip(X['u_id'], X['i_id']):

            pred = self.mean  # if user or item is unknown, we return mean rating by default

            if u_id in self.user_map:
                u_idx = self.user_map[u_id]
                pred += self.user_vector[u_idx]

            if i_id in self.item_map:
                i_idx = self.item_map[i_id]
                pred += self.item_vector[i_idx]

            if u_id in self.user_map and i_id in self.item_map:
                pred += np.dot(self.U[u_idx], self.V[i_idx])

            pred = self.max_rating if pred > self.max_rating else pred
            pred = self.min_rating if pred < self.min_rating else pred
            predictions.append(pred)

        return predictions

    def preprocess_data(self, X, train=True):
        X = X.copy()

        if train:
            user_ids = X['u_id'].unique().tolist()
            item_ids = X['i_id'].unique().tolist()

            self.n_users = len(user_ids)  # total amount of unique users
            self.n_items = len(item_ids)  # same for items

            user_idx = range(self.n_users)
            item_idx = range(self.n_items)

            self.user_map = dict(zip(user_ids, user_idx))  # creating id - index mapping
            self.item_map = dict(zip(item_ids, item_idx))

        X['u_id'] = X['u_id'].map(self.user_map)  # linking users ids to their matrix indices
        X['i_id'] = X['i_id'].map(self.item_map)  # linking items ids to matrix indices

        X.fillna(-1, inplace=True)  # -1 for the missing values

        X['u_id'] = X['u_id'].astype(np.int32)
        X['i_id'] = X['i_id'].astype(np.int32)

        return X[['u_id', 'i_id', 'rating']].values

    def init_metrics(self):
        metrics = np.zeros((self.n_epochs, 3), dtype=np.float)  # creating data with metrics for every epoch
        self.metrics = pd.DataFrame(metrics, columns=['Loss', 'RMSE', 'MAE'])

    def sgd(self, X, X_val):

        user_vector = np.zeros(self.n_users)  # initialize user vector with zeros
        item_vector = np.zeros(self.n_items)

        U = np.random.normal(0, 0.1, (self.n_users, self.n_factors))  # init U and V matrices size (users x factors)
        V = np.random.normal(0, 0.1, (self.n_items, self.n_factors))  # and (items x factors)

        for epoch in range(self.n_epochs):  # Run SGD

            start = time.time()  # measuring time for this epoch
            print('Epoch {}/{}  <3 la '.format(epoch + 1, self.n_epochs))  # do some nice looking print at the begging

            # update U and V on every iteration
            user_vector, item_vector, U, V = self.run_epoch(X, user_vector, item_vector, U, V, self.mean,
                                                            self.n_factors, self.lr, self.reg)

            # compute metrics at this epoch
            self.metrics.loc[epoch,:] = self.compute_metrics(X_val, user_vector, item_vector, U, V, self.mean, self.n_factors)
            # print some nice stuff at the end of the epoch
            self.end_print(start, self.metrics.loc[epoch, 'Loss'], self.metrics.loc[epoch, 'RMSE'], self.metrics.loc[epoch, 'MAE'])

            # stop iterating if RMSE ain't improving by the minimum value of delta
            if epoch > 0:
                if self.metrics.loc[epoch-1, 'RMSE'] - self.min_delta < self.metrics.loc[epoch, 'RMSE']:
                    break

        self.user_vector = user_vector
        self.item_vector = item_vector
        self.U = U
        self.V = V

    def compute_metrics(self, X_val, user_vector, item_vector, U, V, mean, n_factors):

        residuals = []
        for i in range(X_val.shape[0]):
            user, item, rating = X_val[i, 0].astype(np.int32), X_val[i, 1].astype(np.int32), X_val[i, 2]
            pred = mean

            if user != -1:
                pred += user_vector[user]

            if item != -1:
                pred += item_vector[item]

            if user != -1 and item != -1:
                for factor in range(n_factors):
                    pred += U[user, factor] * V[item, factor]

            residuals.append(rating - pred)  # calculate the residuals
        # and get our metrics
        residuals = np.array(residuals)
        loss = np.square(residuals).mean()
        rmse = np.sqrt(loss)
        mae = np.absolute(residuals).mean()
        return loss, rmse, mae

    def run_epoch(self, X, user_vector, item_vector, U, V, global_mean, n_factors, lr, reg):

        for i in range(X.shape[0]):

            user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
            pred = global_mean + user_vector[user] + item_vector[item]  # predict current rating

            for factor in range(n_factors):
                pred += U[user, factor] * V[item, factor]
            err = rating - pred # calc an error

            user_vector[user] += lr * (err - reg * user_vector[user])  # update biases
            item_vector[item] += lr * (err - reg * item_vector[item])

            for factor in range(n_factors):  # update latent factors, using reg coefs and errors
                U_upd = U[user, factor]
                V_upd = V[item, factor]

                U[user, factor] += lr * (err * V_upd - reg * U_upd)
                V[item, factor] += lr * (err * U_upd - reg * V_upd)

        return user_vector, item_vector, U, V

    def end_print(self, start, loss, rmse, mae):
        end = time.time()
        print(f'val_loss: {loss:.2f}', end=', ')
        print(f'val_rmse: {rmse:.2f}', end=', ')
        print(f'val_mae: {mae:.2f}', end=', ')
        print(f'took {end - start:.1f} sec')

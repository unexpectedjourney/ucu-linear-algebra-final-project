import torch 
import random 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from tqdm.cli import tqdm 
from src.metrics import rmse


def set_global_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    
class NeuralSVDModel(torch.nn.Module):
    def __init__(self, k, n_u, n_i): 
        super().__init__()
        self.u_emb = torch.nn.Embedding(n_u, k)        
        self.i_emb = torch.nn.Embedding(n_i, k)

        self.block1 = torch.nn.Sequential(
            torch.nn.Linear(k + k, 1024),
            torch.nn.ReLU(),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(1024),
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(512),
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        
    def forward(self, u_idx, i_idx):
        u_vec = self.u_emb(u_idx)
        i_vec = self.i_emb(i_idx)   
    
        out = torch.cat([u_vec, i_vec], 1)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.head(out)
        out = out.flatten()
        return out


class NeuralSVDModelWithBias(torch.nn.Module):
    def __init__(self, k, n_u, n_i): 
        super().__init__()
        self.u_bias_emb = torch.nn.Embedding(n_u, 1)
        self.i_bias_emb = torch.nn.Embedding(n_i, 1)
        self.neural_svd = NeuralSVDModel(k, n_u, n_i)
        
    def forward(self, u_idx, i_idx):
        u_vec = self.neural_svd.u_emb(u_idx)
        i_vec = self.neural_svd.i_emb(i_idx)   
    
        out = torch.cat([u_vec, i_vec], 1)
        out = self.neural_svd.block1(out)
        out = self.neural_svd.block2(out)
        out = self.neural_svd.block3(out)
        out = self.neural_svd.head(out)
        
        u_bias = self.u_bias_emb(u_idx)
        i_bias = self.i_bias_emb(i_idx)        

        out = out + u_bias + i_bias
        out = out.flatten()
        return out

    
class SimpleSVDModel(torch.nn.Module):
    def __init__(self, k, n_u, n_i): 
        super().__init__()
        self.u_emb = torch.nn.Embedding(n_u, k)        
        self.i_emb = torch.nn.Embedding(n_i, k)
        
    def forward(self, u_idx, i_idx):
        u_vec = self.u_emb(u_idx)
        i_vec = self.i_emb(i_idx)   
        out = torch.bmm(u_vec.unsqueeze(1), i_vec.unsqueeze(2))
        out = out.flatten()
        return out    
    

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_loss = torch.nn.MSELoss()
    
    def forward(self, x1, x2):
        loss = self.base_loss(x1, x2)
        loss = torch.sqrt(loss)
        return loss
    
    
# class CFDataset(torch.utils.data.Dataset):
#     def __init__(
#         self, 
#         df,
#         user_field,
#         item_field,
#         rating_field
#     ):
#         self.df = df
#         self.user_field = user_field
#         self.item_field = item_field
#         self.rating_field = rating_field
    
#     def __getitem__(self, idx):
#         item = self.df.iloc[idx]
#         u = torch.tensor(item[self.user_field])
#         i = torch.tensor(item[self.item_field])
#         r = torch.tensor(item[self.rating_field], dtype=torch.float32)
#         return u, i, r
        
#     def __len__(self):
#         return len(self.df)
    

class CFLoader():
    def __init__(
        self, 
        df,
        user_field,
        item_field,
        rating_field,
        batch_size,
        shuffle=False
    ):
        self.df = df
        # TODO hack
        self.sampler = df
        self.user_field = user_field
        self.item_field = item_field
        self.rating_field = rating_field
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        for idx in range(0, (len(self.df)), self.batch_size):
            select_df = self.df.iloc[idx:idx+self.batch_size]
            yield (
                torch.tensor(select_df[self.user_field].values), 
                torch.tensor(select_df[self.item_field].values),
                torch.tensor(select_df[self.rating_field].values, dtype=torch.float32),
            )
            
            
class NeuralSVD:
    def __init__(
        self,         
        k=50, n_epochs=100, reg=0.02,
        batch_size=1024, lr=0.1,
        model_type=NeuralSVDModel,
        use_scheduler=False,
        scheduler_step=20, scheduler_gamma=0.1,

        
        user_field="customer_id",
        item_field="movie_id",
        target_field="rating",
        random_seed=42,
        verbose=True,
        device=torch.device("cpu"),
    ):
        self.k = k
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.lr = lr
        self.model_type = model_type
        
        self.use_scheduler = use_scheduler
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        
        self.user_field = user_field
        self.item_field = item_field
        self.target_field = target_field
        self.random_seed = random_seed
        self.verbose = verbose
        self.device = device
        
    def fit(self, df, eval_df=None):
        set_global_seed(self.random_seed)
        n_user = len(df[self.user_field].unique())
        n_items = len(df[self.item_field].unique())
        
        self.model = self.model_type(self.k, n_user, n_items)
        self.model = self.model.to(self.device)

        self.criterion = RMSELoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.reg
        )
        
        self.scheduler = None
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.scheduler_step,
                gamma=self.scheduler_gamma
            )

        dataloader = CFLoader(
            df, self.user_field, self.item_field,
            self.target_field, self.batch_size, True
        )
        
        self.train_losses = []
        self.val_scores = []
        
        pbar = tqdm(range(self.n_epochs), desc="train loop", disable=not self.verbose)
        for epoch in pbar:
            train_loss = self.epoch_train(dataloader)
            self.train_losses.append(train_loss)

            if self.scheduler is not None:
                self.scheduler.step()
            lr = [pg for pg in self.optimizer.param_groups][0]["lr"]
            
            desc = f"train loop, loss {train_loss}, lr {lr}"
            
            if eval_df is not None:
                preds = self.predict(eval_df)
                val_score = rmse(eval_df[self.target_field].values, preds)
                self.val_scores.append(val_score)
                desc += f", val rmse {val_score}"

            pbar.set_description(desc)  
        
    def epoch_train(self, dataloader):
        train_loss = 0.0
        self.model.train()
        for batch in dataloader:
            u_idx, i_idx, r = batch

            u_idx = u_idx.to(self.device)
            i_idx = u_idx.to(self.device)
            r = r.to(self.device)

            preds = self.model(u_idx, i_idx)
            loss = self.criterion(preds, r)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            num = r.size(0)
            train_loss += loss.item() * num
        train_loss = train_loss / len(dataloader.sampler)
        return train_loss
    
    def predict(self, df):
        dataloader = CFLoader(
            df, self.user_field, self.item_field, 
            self.target_field, self.batch_size, False
        )
        preds = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader: 
                u_idx, i_idx, r = batch
                u_idx = u_idx.to(self.device)
                i_idx = u_idx.to(self.device)
                r = r.to(self.device)
                preds += self.model(u_idx, i_idx).tolist()

        return preds
    
    def plot(self):
        if len(self.train_losses) > 0:
            plt.plot(self.train_losses, label="train")
        if len(self.val_scores) > 0:
            plt.plot(self.val_scores, label="val")

        plt.xlabel("epoch")
        plt.ylabel("rmse")
        plt.legend()
        plt.grid()

        
## Warning - то може бути шкідливо для оченят ##
# Model for rating classification (not just a regression) 
# class NeuralSVDModelClassify(torch.nn.Module):
#     def __init__(self, k, n_u, n_i): 
#         super().__init__()
#         self.u_emb = torch.nn.Embedding(n_u, k)        
#         self.i_emb = torch.nn.Embedding(n_i, k)
#         self.block1 = torch.nn.Sequential(
#             torch.nn.Linear(k + k, 1024),
#             torch.nn.ReLU(),
#         )        
#         self.block2 = torch.nn.Sequential(
#             torch.nn.BatchNorm1d(1024),
#             torch.nn.Dropout(0.2),
#             torch.nn.Linear(1024, 512),
#             torch.nn.ReLU(),
#         )       
#         self.block3 = torch.nn.Sequential(
#             torch.nn.BatchNorm1d(512),
#             torch.nn.Dropout(0.2),
#             torch.nn.Linear(512, 128),
#             torch.nn.ReLU(),
#         )
#         self.head = torch.nn.Sequential(
#             torch.nn.Linear(128, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 5)
#         )
        
        
#     def forward(self, u_idx, i_idx):
#         u_vec = self.u_emb(u_idx)
#         i_vec = self.i_emb(i_idx)   
#         out = torch.cat([u_vec, i_vec], 1)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.head(out)
# #         out = out.flatten()
#         return out

# class CFLoaderClassify():
#     def __init__(
#         self, 
#         df,
#         user_field,
#         item_field,
#         rating_field,
#         batch_size,
#         shuffle=False
#     ):
#         self.df = df
#         # TODO hack
#         self.sampler = df
#         self.user_field = user_field
#         self.item_field = item_field
#         self.rating_field = rating_field
#         self.batch_size = batch_size
#         self.shuffle = shuffle

#     def __iter__(self):
#         if self.shuffle:
#             self.df = self.df.sample(frac=1).reset_index(drop=True)
#         for idx in range(0, (len(self.df)), self.batch_size):
#             select_df = self.df.iloc[idx:idx+self.batch_size]
#             yield (
#                 torch.tensor(select_df[self.user_field].values), 
#                 torch.tensor(select_df[self.item_field].values),
# #                 torch.tensor(select_df[self.rating_field].values, dtype=torch.float32),
# #                 torch.tensor(select_df[self.rating_field].values).long() - 1
#                 torch.nn.functional.one_hot(torch.tensor(select_df[self.rating_field].values).long() - 1).float()
#             )

# set_global_seed()

# batch_size = 1024
# dataloaders = {}
# dataloaders["train"] = CFLoaderClassify(tr_df, "customer_id", "movie_id", "rating", batch_size, True)
# dataloaders["val"] = CFLoaderClassify(val_df, "customer_id", "movie_id", "rating", batch_size, False)

# # criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.BCEWithLogitsLoss()
# model = NeuralSVDModelClassify(100, 1000, 1000)
# # model = NeuralSVDModelWithBias(50, 1000, 1000)
# # optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.0)
# # optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.001)
# # scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
# scheduler = None

# model = model.to(device)

# n_epochs = 100
# dataloader = dataloaders["train"]
# model.train()

# losses = []
# pbar = tqdm(range(n_epochs), desc="train loop")
# for epoch in pbar:
#     train_loss = 0.0
#     for batch in dataloader:
#         u_idx, i_idx, r = batch
#         u_idx = u_idx.to(device)
#         i_idx = u_idx.to(device)
#         r = r.to(device)      
#         preds = model(u_idx, i_idx)
#         loss = criterion(preds, r)
# #         loss = torch.sqrt(loss)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()  
#         num = r.size(0)
#         train_loss += loss.item() * num        
#     train_loss = train_loss / len(dataloader.sampler)
#     losses.append(train_loss)
#     if scheduler is not None:
#             scheduler.step()
#     lr = [pg for pg in optimizer.param_groups][0]["lr"]
#     pbar.set_description(f"train loop, loss {train_loss}, lr {lr}")

# preds = []
# dataloader = dataloaders["val"]
# model.eval()
# with torch.no_grad():
#     for batch in dataloader: 
#         u_idx, i_idx, r = batch
#         u_idx = u_idx.to(device)
#         i_idx = u_idx.to(device)
#         r = r.to(device)
#         preds += (model(u_idx, i_idx).argmax(1) + 1).tolist()
# #         break
# rmse(val_df.rating.values, preds)  

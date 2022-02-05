#!/usr/bin/env python
# coding: utf-8

# In[99]:


import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
import yaml
from deepchoice.data import ChoiceDataset
from deepchoice.data.utils import create_data_loader
from deepchoice.model.bemb_flex_lightning import LitBEMBFlex
from sklearn.preprocessing import LabelEncoder
from termcolor import cprint

from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from deepchoice.model import BEMBFlex

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import yaml
from deepchoice.data import ChoiceDataset
from deepchoice.data.utils import create_data_loader
from deepchoice.model.bemb_flex_lightning import LitBEMBFlex
from sklearn.preprocessing import LabelEncoder
from termcolor import cprint

from deepchoice.model import BEMBFlex

import seaborn as sns


# # The Simulation Study for Quick Checks on the Correctness of Algorithm: Define the data-generating-process (DGP)
#
# $$
# U_{uit} = \theta_i^\top \beta_u P
# $$
#
# $$
# \theta_i \sim \mathcal{N}
# $$
#
# $$
# \beta_u \sim \mathcal{N}
# $$

# In[121]:


NUM_USERS = 5
NUM_ITEMS = 20
NUM_SESSIONS = 10
K = 5
N = 1000
NUM_P_DIM = 3


# In[122]:


user_mean = torch.randn((NUM_USERS, K * NUM_P_DIM))
item_mean = - torch.randn((NUM_ITEMS, K * NUM_P_DIM))


# In[123]:


theta_dist = MultivariateNormal(loc=item_mean, covariance_matrix=torch.eye(K * NUM_P_DIM))
theta = theta_dist.sample()
theta = theta.reshape(NUM_ITEMS, K, NUM_P_DIM)

beta_dist = MultivariateNormal(loc=user_mean, covariance_matrix=torch.eye(K * NUM_P_DIM))
beta = beta_dist.sample()
beta = beta.reshape(NUM_USERS, K, NUM_P_DIM)


# In[124]:


user_index = torch.LongTensor(np.random.choice(range(NUM_USERS), size=N, replace=True))
session_index = torch.LongTensor(np.random.choice(range(NUM_SESSIONS), size=N, replace=True))


# In[125]:


P = torch.randn(size=(NUM_SESSIONS, NUM_ITEMS, NUM_P_DIM))


# In[126]:


sns.displot(user_mean.reshape(-1,))


# ## Compute Utilities and Generate Labels

# In[127]:


utility_array = list()
for row in tqdm(range(N)):
    u = user_index[row]
    t = session_index[row]
    utility = torch.zeros(NUM_ITEMS)
    for i in range(NUM_ITEMS):
        theta_i = theta[i]
        beta_u = beta[u]
        coef = (theta_i * beta_u).sum(dim=0)
        price = P[t, i, :]
        utility[i] = (coef * price).sum()
    utility_array.append(utility)


# In[128]:


U = torch.stack(utility_array)
label = torch.argmax(U, dim=1)


# # Build Dataset

# In[129]:


train_mask = np.arange(0, int(0.7*N))
val_mask = np.arange(int(0.7*N), int(0.85*N))
test_mask = np.arange(int(0.85*N), N)

dataset_list = list()
for mask in (train_mask, val_mask, test_mask):
    d = ChoiceDataset(label=label[mask],
                      user_index=user_index[mask],
                      session_index=session_index[mask],
                      item_availability=None,
                      price_obs=P)
    dataset_list.append(d)


# In[130]:


class LitBEMBFlex(pl.LightningModule):
    def __init__(self, **kwargs):
        # use kwargs to pass parameter to BEMB Torch.
        super().__init__()
        self.model = BEMBFlex(**kwargs)
        self.num_needs = 4
        self.learning_rate = 0.3
        self.batch_size = 500

    def __str__(self) -> str:
        return str(self.model)

    def training_step(self, batch, batch_idx):
        elbo = self.model.elbo(batch, num_seeds=self.num_needs)
        self.log('train_elbo', elbo)
        loss = - elbo
        return loss

    def validation_step(self, batch, batch_idx):
        LL = self.model.forward(
            batch, return_logit=False, all_items=False).mean()
        self.log('val_log_likelihood', LL, prog_bar=True)

    def test_step(self, batch, batch_idx):
        LL = self.model.forward(
            batch, return_logit=False, all_items=False).mean()
        self.log('test_log_likelihood', LL)

        pred = self.model(batch)
        performance = self.model.get_within_category_accuracy(
            pred, batch.label)
        for key, val in performance.items():
            self.log('test_' + key, val, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return create_data_loader(dataset_list[0], batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return create_data_loader(dataset_list[1], batch_size=self.batch_size, shuffle=True, num_workers=8)

    def test_dataloader(self):
        # use smaller batch size for test, which takes more memory.
        return create_data_loader(dataset_list[2], batch_size=10000, shuffle=False, num_workers=8)


# In[131]:


num_latents = 50

model = LitBEMBFlex(utility_formula='theta_item * beta_user * price_obs',
                    num_users=NUM_USERS,
                    num_items=NUM_ITEMS,
                    num_sessions=NUM_SESSIONS,
                    obs2prior_dict={'theta_item': False, 'beta_user': False},
                    coef_dim_dict={'theta_item': num_latents * NUM_P_DIM, 'beta_user': num_latents * NUM_P_DIM},
                    num_price_obs=NUM_P_DIM)
trainer = pl.Trainer(
    max_epochs=10,
    check_val_every_n_epoch=1,
    log_every_n_steps=1,
    gpus=1)
trainer.fit(model)


# In[132]:


trainer.test(model)


# In[133]:
quit()

num_epochs = 50

callback = TuneReportCallback({'val_log_likelihood': 'val_log_likelihood'}, on='validation_end')

def train_tune(hparams, epochs=10, gpus=1):
    model = LitBEMBFlex(utility_formula='theta_item * beta_user * price_obs',
                    num_users=NUM_USERS,
                    num_items=NUM_ITEMS,
                    num_sessions=NUM_SESSIONS,
                    obs2prior_dict={'theta_item': False, 'beta_user': False},
                    coef_dim_dict={'theta_item': hparams['num_latents'] * NUM_P_DIM, 'beta_user': hparams['num_latents'] * NUM_P_DIM},
                    num_price_obs=NUM_P_DIM)

    trainer = pl.Trainer(
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        gpus=gpus,
        progress_bar_refresh_rate=0,
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name='', version='.'),
        callbacks=[callback])
    trainer.fit(model)


# In[ ]:


num_samples = 20

config = {
    'num_latents': tune.choice([1, 2, 3, 4, 5, 7, 10, 20, 50, 100, 300, 1000])
}

scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

reporter = CLIReporter(parameter_columns=list(config.keys()),
                       metric_columns=list(callback._metrics.keys()))

analysis = tune.run(
    tune.with_parameters(train_tune, epochs=num_epochs, gpus=1),
    metric='val_log_likelihood',
    mode='max',
    resources_per_trial={'cpu': 16, 'gpu': 1},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter)

print("Best hyperparameters found were: ", analysis.best_config)


# In[ ]:





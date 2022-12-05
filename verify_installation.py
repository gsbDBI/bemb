"""
You can use this script to verify that your installation of BEMB. This script generates a small dataset and runs a trail training on it.
"""
import numpy as np
import pandas as pd
import torch
from torch_choice.data import ChoiceDataset
from bemb.model import LitBEMBFlex
from bemb.utils.run_helper import run
import matplotlib.pyplot as plt
import seaborn as sns

# simulate dataset
num_users = 1500
num_items = 50
data_size = 1000

user_index = torch.LongTensor(np.random.choice(num_users, size=data_size))
Us = np.arange(num_users)
Is = np.sin(np.arange(num_users) / num_users * 4 * np.pi)
Is = (Is + 1) / 2 * num_items
Is = Is.astype(int)

PREFERENCE = dict((u, i) for (u, i) in zip(Us, Is))

# construct users.
item_index = torch.LongTensor(np.random.choice(num_items, size=data_size))

for idx in range(data_size):
    if np.random.rand() <= 0.5:
        item_index[idx] = PREFERENCE[int(user_index[idx])]

user_obs = torch.zeros(num_users, num_items)
user_obs[torch.arange(num_users), Is] = 1

item_obs = torch.eye(num_items)

dataset = ChoiceDataset(
    user_index=user_index,
    item_index=item_index,
    user_obs=user_obs,
    item_obs=item_obs)

idx = np.random.permutation(len(dataset))
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
train_idx = idx[:train_size]
val_idx = idx[train_size: train_size + val_size]
test_idx = idx[train_size + val_size:]

dataset_list = [dataset[train_idx], dataset[val_idx], dataset[test_idx]]

bemb = LitBEMBFlex(
    # set the learning rate, feel free to play with different levels.
    learning_rate=0.03,
    pred_item=True,  # let the model predict item_index, don't change this one.
    num_seeds=32,  # number of Monte Carlo samples for estimating the ELBO.
    utility_formula='theta_user * alpha_item',  # the utility formula.
    num_users=num_users,
    num_items=num_items,
    num_user_obs=dataset.user_obs.shape[1],
    num_item_obs=dataset.item_obs.shape[1],
    # whether to turn on obs2prior for each parameter.
    obs2prior_dict={'theta_user': True, 'alpha_item': True},
    # the dimension of latents, since the utility is an inner product of theta and alpha, they should have
    # the same dimension.
    coef_dim_dict={'theta_user': 10, 'alpha_item': 10}
)

bemb = bemb.to('cuda')

# use the provided run helper to train the model.
# we set batch size to be 5% of the data size, and train the model for 50 epochs.
# there would be 20*50=1,000 gradient update steps in total.
bemb = bemb.fit_model(
    dataset_list,
    batch_size=len(dataset) // 20,
    num_epochs=50)

print("Successful!")

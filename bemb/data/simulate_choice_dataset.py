"""
Generate a simulated choice dataset for tutorials, unit tests, and debugging.
"""
from typing import List

import numpy as np
import torch
from torch_choice.data import ChoiceDataset


def load_simulation_dataset(num_users: int, num_items: int, data_size: int) -> List[ChoiceDataset]:
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

    dataset = ChoiceDataset(user_index=user_index, item_index=item_index, user_obs=user_obs, item_obs=item_obs)

    idx = np.random.permutation(len(dataset))
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_idx = idx[:train_size]
    val_idx = idx[train_size: train_size + val_size]
    test_idx = idx[train_size + val_size:]

    dataset_list = [dataset[train_idx], dataset[val_idx], dataset[test_idx]]
    return dataset_list

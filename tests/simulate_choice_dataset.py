"""
Generate a simulated choice dataset for tutorials, unit tests, and debugging.
"""
from typing import List

import numpy as np
import torch
from torch_choice.data import ChoiceDataset


def simulate_dataset(num_users: int, num_items: int, data_size: int) -> List[ChoiceDataset]:
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


def simulate_dataset_v2(num_users: int, num_items: int, num_sessions: int, data_size: int) -> List[ChoiceDataset]:
    length_of_dataset = data_size  # $N$
    # create observables/features, the number of parameters are arbitrarily chosen.
    # generate 128 features for each user, e.g., race, gender.
    # these variables should have shape (num_users, *)
    user_obs = torch.randn(num_users, 128)
    # generate 64 features for each user, e.g., quality.
    item_obs = torch.randn(num_items, 64)
    # generate 10 features for each session, e.g., weekday indicator.
    session_obs = torch.randn(num_sessions, 10)
    # generate 12 features for each session user pair, e.g., the budget of that user at the shopping day.
    itemsession_obs = torch.randn(num_sessions, num_items, 12)
    # generate 12 features for each user item pair, e.g., the user's preference on that item.
    useritem_obs = torch.randn(num_users, num_items, 12)
    # generate 10 user-session specific observables, e.g., the historical spending amount of that user at that session.
    usersession_obs = torch.randn(num_users, num_sessions, 10)
    # generate 8 features for each user session item triple, e.g., the user's preference on that item at that session.
    # since `U*S*I` is potentially huge and may cause identifiability issues, we rarely use this kind of observable in practice.
    usersessionitem_obs = torch.randn(num_users, num_sessions, num_items, 8)

    # generate the array of item[n].
    item_index = torch.LongTensor(np.random.choice(num_items, size=length_of_dataset))
    # generate the array of user[n].
    user_index = torch.LongTensor(np.random.choice(num_users, size=length_of_dataset))
    # generate the array of session[n].
    session_index = torch.LongTensor(np.random.choice(num_sessions, size=length_of_dataset))

    # assume all items are available in all sessions.
    item_availability = torch.ones(num_sessions, num_items).bool()

    dataset = ChoiceDataset(
        # pre-specified keywords of __init__
        item_index=item_index,  # required.
        num_items=num_items,
        # optional:
        user_index=user_index,
        num_users=num_users,
        session_index=session_index,
        item_availability=item_availability,
        # additional keywords of __init__
        user_obs=user_obs,
        item_obs=item_obs,
        session_obs=session_obs,
        itemsession_obs=itemsession_obs,
        useritem_obs=useritem_obs,
        usersession_obs=usersession_obs,
        usersessionitem_obs=usersessionitem_obs)

    # we can subset the dataset by conventional python indexing.
    dataset_train = dataset[:int(0.8*len(dataset))]
    dataset_val = dataset[int(0.8*len(dataset)):int(0.9*len(dataset))]
    dataset_test = dataset[int(0.9*len(dataset)):]

    return [dataset_train, dataset_val, dataset_test]

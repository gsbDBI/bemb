import argparse
import time
import os

import deepchoice
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from deepchoice.data import ChoiceDataset
from deepchoice.data.utils import create_data_loader
from deepchoice.model import BEMB
from termcolor import cprint
from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder

DEVICE = 'cuda'
NUM_EPOCHS = 100

args = argparse.Namespace(shuffle=True, batch_size=256)

# PREFERENCE = (
#     (0, 3),
#     (1, 16),
#     (4, 5),
#     (3, 6)
# )

if __name__ == '__main__':
    cprint('Your are running a debugging script!', 'red')

    num_users = 150
    num_items = 50

    # some artifical preference patterns for debugging.
    # item_liked_by(u) = sin(u).
    Us = np.arange(num_users)
    Is = np.sin(np.arange(num_users) / num_users * 2 * np.pi)
    Is = (Is + 1) / 2 * num_items
    Is = Is.astype(int)

    PREFERENCE = ((u, i) for (u, i) in zip(Us, Is))

    # for (u, i) in zip(Us, Is):
    #     MANIPULATION.append((u, int(i)))

    category_to_item = {0: torch.arange(num_items // 2) * 2,
                        1: torch.arange(num_items // 2) * 2 + 1}
    num_sessions = 10000
    dim = 5  # embedding dimension.

    # user_features = torch.randn(num_users, 1024)
    # item_features = torch.randn(num_items, 512)
    session_features = torch.randn(num_sessions)

    user_onehot = torch.zeros(num_sessions, num_users)
    user_idx = torch.LongTensor(np.random.choice(num_users, size=num_sessions))
    user_onehot[torch.arange(num_sessions), user_idx] = 1

    # different user characteristics.
    label = torch.LongTensor(np.random.choice(num_items, size=num_sessions))

    # add some artificial pattern.
    # user WHO loves item LIKES especially.
    for WHO, LIKES in PREFERENCE:
        label[torch.logical_and(user_idx == WHO, torch.rand(num_sessions) <= 0.9)] = LIKES

    dataset = ChoiceDataset(label=label, user_onehot=user_onehot,
                            item_availability=None).to(DEVICE)
    dataloader = create_data_loader(dataset, args)

    model = BEMB(num_users, num_items, dim, False, category_to_item).to(DEVICE)

    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for i in range(NUM_EPOCHS):
        total_loss = torch.scalar_tensor(0.0).to(DEVICE)
        for batch in dataloader:
            # maximize the ELBO.
            loss = - model.elbo(batch, num_seeds=1024)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
        if i % (NUM_EPOCHS // 10) == 0:
            print(f'Epoch [{i}] negative elbo (the lower the better) = {total_loss}')
    print(f'Time taken: {time.time() - start_time: 0.1f} seconds.')

    user_latent = model.user_latent_q.mean(torch.eye(num_users).to(DEVICE))  # (num_users, dim)
    item_latent = model.item_latent_q.mean(torch.eye(num_items).to(DEVICE))  # (num_items, dim)
    affinity = user_latent @ item_latent.T  # (num_users, num_items)
    affinity = affinity.detach().cpu().numpy()

    ax = sns.heatmap(affinity, square=False)
    ax.set_ylabel('user ID')
    ax.set_xlabel('item ID')
    ax.set_title('inner product of user embedding and item embedding')
    plt.savefig('./out.png')

    cprint(f'{model.num_params=:}, matrix size={num_items * num_users}', 'green')
    cprint('Done.', 'green')

    breakpoint()

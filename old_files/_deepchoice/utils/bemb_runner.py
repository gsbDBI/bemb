"""
A helper script to run PyTorch BEMB model using input files from previous BEMB in C++.
We make the argument names as close to the original C++ implementation as possible.
"""
import argparse
import os
import time

import _deepchoice
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from _deepchoice.data import ChoiceDataset
from _deepchoice.data.utils import create_data_loader
from _deepchoice.model import BEMB
from sklearn.preprocessing import LabelEncoder
from termcolor import cprint
from torch._C import parse_schema
from torch.utils.data import DataLoader


def make_arg_parser():
    parser = argparse.ArgumentParser()
    # args from C++ implementation.
    parser.add_argument('--dir', type=str, help='path to the data folder')
    parser.add_argument('--outdir', type=str, help='path to the output folder', default='./')
    parser.add_argument('--K', type=int, help='number of latent factors', default=10)
    parser.add_argument('--max_iterations ', type=int, helper='maximum number of iterations', default=100)
    parser.add_argument('--rfreq', type=int, help='the test log-lik will be computed every `rfreq` iterations',
                        default=5)
    parser.add_argument('--eta', type=float, help='stepsize parameter', default=0.03)
    parser.add_argument('--batchsize', type=int, help='number of datapoints per batch', default=-1)

    parser.add_argument('--UC', type=int, help='incorporate user observables? (0=no, integer=number of observables)',
                        default=0)
    parser.add_argument('--IC', type=int, help='incorporate item observables? (0=no, integer=number of observables)',
                        default=0)

    parser.add_argument('--obs2prior_user', type=bool, default=False)
    parser.add_argument('--obs2prior_item', type=bool, default=False)

    parser.add_argument('--obs2utility_user', type=bool, default=False)
    parser.add_argument('--obs2utility_item', type=bool, default=False)
    parser.add_argument('--obs2utility_session', type=bool, default=False)
    parser.add_argument('--obs2utility_taste', type=bool, default=False)
    parser.add_argument('--obs2utility_price', type=bool, default=False)

    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--likelihood', type=str, default='within_category')

    parser.add_argument('--device', type=str, default='cpu')
    return parser


if __name__ == '__main__':
    cprint('Your are running a debugging script!', 'red')

    parser = make_arg_parser()
    args = parser.parse_args()

    # for debugging.
    data_dir = '/home/tianyudu/Data/MoreSupermarket/tsv'

    def is_sorted(x):
        return all(x == np.sort(x))

    def load_tsv(file_name):
        return pd.read_csv(os.path.join(data_dir, file_name),
                           sep='\t',
                           index_col=None,
                           names=['user_id', 'item_id', 'session_id', 'quantity'])

    train = load_tsv('train.tsv')
    validation = load_tsv('validation.tsv')
    test = load_tsv('test.tsv')

    data_all = pd.concat([train, validation, test], axis=0)
    num_sessions = len(data_all)

    # TODO: improve efficiency.
    # num_users = data_all['user_id'].nunique()
    # num_items = data_all['item_id'].nunique()

    # ordinal encoding users and items.
    user_encoder = LabelEncoder().fit(data_all['user_id'].values)
    num_users = len(user_encoder.classes_)
    assert is_sorted(user_encoder.classes_)
    item_encoder = LabelEncoder().fit(data_all['item_id'].values)
    num_items = len(item_encoder.classes_)
    assert is_sorted(item_encoder.classes_)

    # load observables.
    user_obs = pd.read_csv(os.path.join(data_dir, 'obsUser.tsv'), sep='\t', index_col=0, header=None)
    user_obs = user_obs.groupby(user_obs.index).first().sort_index()
    num_user_features = user_obs.shape[1]
    user_obs = torch.Tensor(user_obs.values)

    item_obs = pd.read_csv(os.path.join(data_dir, 'obsItem.tsv'), sep='\t', index_col=0, header=None)
    item_obs = item_obs.groupby(item_obs.index).first().sort_index()
    num_item_features = item_obs.shape[1]
    item_obs = torch.Tensor(item_obs.values)

    dataset_list = list()
    for d in (train, validation, test):
        user_idx = user_encoder.transform(d['user_id'].values)

        user_onehot = torch.zeros(len(d), num_users)
        user_onehot[torch.arange(len(d)), user_idx] = 1
        user_onehot = user_onehot.long()

        label = torch.LongTensor(item_encoder.transform(d['item_id'].values))

        choice_dataset = ChoiceDataset(label=label, user_onehot=user_onehot,
                                       item_availability=None,
                                       user_features=user_obs,
                                       item_features=item_obs)
        dataset_list.append(choice_dataset)

    # ... Load obs...
    item_groups = pd.read_csv(os.path.join(data_dir, 'itemGroup.tsv'),
                              sep='\t',
                              index_col=None,
                              names=['item_id', 'category_id'])

    # TODO: fix.
    item_groups = item_groups.groupby('item_id').first().reset_index()
    # filter out items not in any dataset above.
    mask = item_groups['item_id'].isin(item_encoder.classes_)
    item_groups = item_groups[mask].reset_index(drop=True)
    item_groups = item_groups.sort_values(by='item_id')

    category_encoder = LabelEncoder().fit(item_groups['category_id'])
    num_categories = len(category_encoder.classes_)

    item_groups['item_id'] = item_encoder.transform(
        item_groups['item_id'].values)
    item_groups['category_id'] = category_encoder.transform(
        item_groups['category_id'].values)

    item_groups = item_groups.groupby('category_id')['item_id'].apply(list)
    category_to_item = dict(zip(item_groups.index, item_groups.values))

    dim = K  # embedding dimension.

    # dataloaders = [create_data_loader(dataset, args) for dataset in dataset_list]
    # only care the training set for now.
    # dataloader = dataloaders[0]
    dataset = dataset_list[0].to(DEVICE)
    dataloader = create_data_loader(dataset, args)

    model = BEMB(num_users=num_users,
                 num_items=num_items,
                 latent_dim=dim,
                 trace_log_q=False,
                 category_to_item=category_to_item,
                 likelihood='within_category',
                 obs2prior_user=True,
                 obs2prior_item=True,
                 num_user_features=num_user_features,
                 num_item_features=num_item_features).to(DEVICE)

    start_time = time.time()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=1)

    # breakpoint()
    # print(model.user_latent_q.mean.weight.T)
    # model.load_params('/home/tianyudu/Data/MoreSupermarket/t79338-n2123-m1109-k3-users3-lik3-shuffle0-eta0.005-zF0.1-nS10-batch5000-run_K3_500')
    # print(model.user_latent_q.mean.weight.T)

    print(80 * '=')
    print(model)
    print(80 * '=')

    for i in range(NUM_EPOCHS):
        total_loss = torch.scalar_tensor(0.0).to(DEVICE)

        for batch in dataloader:
            # maximize the ELBO.
            loss = - model.elbo(batch, num_seeds=4)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()

        # if i % (NUM_EPOCHS // 10) == 0:
        scheduler.step()
        if True:
            print(model.report_performance())
            print(
                f'Epoch [{i}] negative elbo (the lower the better)={total_loss}')
    print(f'Time taken: {time.time() - start_time: 0.1f} seconds.')

    quit()

    print(model.user_latent_q.mean.weight.T)

    user_latent = model.user_latent_q.mean.weight.T  # (num_users, dim)
    item_latent = model.item_latent_q.mean.weight.T  # (num_items, dim)
    affinity = user_latent @ item_latent.T  # (num_users, num_items)
    torch.save(affinity, './output/similarity.pt')
    affinity = affinity.detach().cpu().numpy()

    torch.save(user_latent, './output/theta.pt')
    torch.save(item_latent, './output/alpha.pt')

    ax = sns.heatmap(affinity, square=False)
    ax.set_ylabel('user ID')
    ax.set_xlabel('item ID')
    ax.set_title('inner product of user embedding and item embedding')
    plt.savefig('./out.png')

    cprint('Done.', 'green')
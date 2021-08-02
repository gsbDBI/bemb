"""
Test consistency of the dataloader module.
"""
import os
import sys
sys.path.append('./')

import pandas as pd
import torch
from data.data_formatter import stata_to_tensors
from data.dataset import CMDataset
from termcolor import colored
from torch.utils.data.sampler import (BatchSampler, RandomSampler,
                                      SequentialSampler)


def main():
    project_path = '~/Development/deepchoice'
    mode_canada = pd.read_csv(os.path.join(project_path, 'data/ModeCanada.csv'), index_col=0)
    mode_canada = mode_canada.query('noalt == 4').reset_index(drop=True)

    print(mode_canada.shape)
    # u: user, i: item, t: time/session.
    var_cols_dict = {
        'u': None,
        'i': ['cost', 'freq', 'ovt'],
        'ui': None,
        't': ['income'],
        'ut': None,
        'it': ['ivt'],
        'uit': None
    }
    # make the dataset from long format.
    X, user_onehot, A, Y, C = stata_to_tensors(df=mode_canada,
                                               var_cols_dict=var_cols_dict,
                                               session_id='case',
                                               item_id='alt',
                                               choice='choice',
                                               category_id=None,
                                               user_id=None)

    for k, v in X.items():
        if torch.is_tensor(v):
            print(f'X[{k}].shape = {v.shape}')

    dataset = CMDataset(X=X, user_onehot=user_onehot, A=A, Y=Y, C=C)
    # check for full-batch loading results.
    sampler = BatchSampler(
        SequentialSampler(dataset),
        batch_size=len(dataset), drop_last=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             sampler=sampler,
                                             num_workers=0,
                                             collate_fn=lambda x: x[0])

    features, label = next(iter(dataloader))
    X_all, user_onehot_all, A_all, C_all = features
    y_all = label
    for key, val in X.items():
        if val is not None:
            assert torch.all(X[key] == X_all[key])
    assert torch.all(user_onehot_all == user_onehot)
    assert torch.all(Y == y_all)
    assert torch.all(A == A_all)
    assert torch.all(C == C_all)

    # check mini-batches.
    del dataloader

    def select_batch(input, idx):
        out = dict()
        for key, val in input.items():
            if val is None:
                out[key] = None
            else:
                out[key] = input[key][idx]
        return out

    for batch_size in [1, 128, 1024]:
        sampler = BatchSampler(
            SequentialSampler(dataset),
            batch_size=batch_size, drop_last=False)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 sampler=sampler,
                                                 num_workers=0,
                                                 collate_fn=lambda x: x[0])

        for i, (features, label) in enumerate(dataloader):
            X_batch, user_onehot_batch, A_batch, C_batch = features
            y_batch = label
            # check consistency.
            first, last = i * batch_size, min(len(Y), (i + 1) * batch_size)
            idx = torch.arange(first, last)
            X_expect = select_batch(X, idx)
            for key, val in X_expect.items():
                if val is not None:
                    assert torch.all(X_expect[key] == X_batch[key])
            assert torch.all(user_onehot_batch == user_onehot[idx])
            assert torch.all(Y[idx] == y_batch)
            assert torch.all(A[idx] == A_batch)
            assert torch.all(C[idx] == C_batch)


if __name__ == '__main__':
    main()
    print(colored('Passed!', 'green'))

"""
Test consistency of the dataloader module.
"""
from train.train import train
from model.conditional_logit_model import ConditionalLogitModel
from data.dataloader import CMDataset, collate_fn
from data.data_formatter import stata_to_tensors
from termcolor import colored
import argparse
import os
import pdb
import sys
from typing import Optional

import pandas as pd
import torch
import yaml

sys.path.append('./')


def main(modifier: Optional[dict] = None):
    arg_path = './args_ModeCanada.yaml'
    with open(arg_path) as file:
        # unwrap dictionary loaded.
        args = argparse.Namespace(**yaml.load(file, Loader=yaml.FullLoader))

    if modifier is not None:
        for k, v in modifier.items():
            setattr(args, k, v)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # data.
    project_path = '~/Development/deepchoice'
    mode_canada = pd.read_csv(os.path.join(
        project_path, 'data/ModeCanada.csv'), index_col=0)
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
    # X, user_onehot, A, Y, C = stata_to_tensors(df=mode_canada,
    #                                            var_cols_dict=var_cols_dict,
    #                                            session_id='case',
    #                                            item_id='alt',
    #                                            choice='choice',
    #                                            category_id=None,
    #                                            user_id=None)

    X, user_onehot, A, Y, C = torch.load('./ModeCanada.pt')

    for k, v in X.items():
        if torch.is_tensor(v):
            print(f'X[{k}].shape = {v.shape}')

    dataset = CMDataset(X=X, user_onehot=user_onehot, A=A, Y=Y, C=C)
    # check for full-batch loading results.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False,
                                             collate_fn=collate_fn)

    features, label = next(iter(dataloader))
    X_all, user_onehot_all, A_all, C_all = features
    y_all = label
    for key, val in X.items():
        if val is not None:
            assert torch.all(X[key] == X_all[key])
    assert torch.all(user_onehot_all == user_onehot)
    assert torch.all(Y == y_all)

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

    batch_size = 128
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
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


if __name__ == '__main__':
    main()
    print(colored('Passed!', 'green'))

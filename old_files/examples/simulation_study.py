import argparse
import os
import pdb
import sys
from typing import Optional

import pandas as pd
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
import yaml

sys.path.append('../src')
sys.path.append('./')
from data.data_formatter import stata_to_tensors
from data.dataset import CMDataset
from data.simulation_dataset import generate_dataset
from model.conditional_logit_model import ConditionalLogitModel
from train.train import train


if __name__ == '__main__':
    arg_path = sys.argv[1]
    with open(arg_path) as file:
        args = argparse.Namespace(**yaml.load(file, Loader=yaml.FullLoader))  # unwrap dictionary loaded.

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    breakpoint()

    num_sessions = 1000
    
    X, user_onehot, Y = generate_dataset(num_users=args.num_users,
                                         num_user_features=args.var_num_params_dict['u'],
                                         num_items=args.num_items,
                                         num_item_features=args.var_num_params_dict['i'],
                                         num_sessions=num_sessions)
    A = torch.ones((num_sessions, args.num_items)).bool()
    C = torch.ones(num_sessions).long()

    dataset = CMDataset(X=X, user_onehot=user_onehot, A=A, Y=Y, C=C, device='cuda')

    data_train = torch.utils.data.DataLoader(dataset,
                                             sampler=BatchSampler(RandomSampler(dataset),
                                                                  batch_size=128,
                                                                  drop_last=False),
                                             num_workers=0 if dataset.device == 'cuda' else os.cpu_count(),
                                             collate_fn=lambda x: x[0],
                                             pin_memory=(dataset.device == 'cpu'))

    model = ConditionalLogitModel(num_items=args.num_items,
                                  num_users=args.num_users,
                                  var_variation_dict=args.var_variation_dict,
                                  var_num_params_dict=args.var_num_params_dict)

    model = model.to(args.device)

    train([data_train, None, None], model, args)

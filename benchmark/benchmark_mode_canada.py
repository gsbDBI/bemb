"""
Benchmark using the ModeCanada dataset in R.
"""
import argparse
import os
import pdb
import sys
from typing import Optional

import pandas as pd
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
import yaml

sys.path.append('./src')
sys.path.append('./')
from data.data_formatter import stata_to_tensors
from data.dataset import CMDataset
from model.conditional_logit_model import ConditionalLogitModel
from train.train import train


def main(modifier: Optional[dict]=None):
    arg_path = sys.argv[1]
    # arg_path = './args.yaml'
    with open(arg_path) as file:
        args = argparse.Namespace(**yaml.load(file, Loader=yaml.FullLoader))  # unwrap dictionary loaded.
    
    if modifier is not None:
        for k, v in modifier.items():
            setattr(args, k, v)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    project_path = '~/Development/deepchoice'
    mode_canada = pd.read_csv(os.path.join(project_path, 'data/ModeCanada.csv'), index_col=0)
    mode_canada = mode_canada.query('noalt == 4').reset_index(drop=True)

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
    
    # torch.save((X, user_onehot, A, Y, C), './ModeCanada.pt')
    X, user_onehot, A, Y, C = torch.load('./ModeCanada.pt')

    model = ConditionalLogitModel(num_items=args.num_items,
                                  num_users=args.num_users,
                                  var_variation_dict=args.var_variation_dict,
                                  var_num_params_dict=args.var_num_params_dict)

    model = model.to(args.device)
    print(f'Number of parameters: {model.total_params}')
    dataset = CMDataset(X=X, user_onehot=user_onehot, A=A, Y=Y, C=C, device='cuda')
    if args.batch_size == -1:
        # use full-batch.
        args.batch_size = len(dataset)

    sampler = BatchSampler(
        RandomSampler(dataset) if args.shuffle else SequentialSampler(dataset),
        batch_size=args.batch_size,
        drop_last=False)
    # feed a batch_sampler as sampler so that dataset.__getitem__ is called with a list of indices.
    # cannot use multiple workers if the entire dataset is already on GPU.
    data_train = torch.utils.data.DataLoader(dataset,
                                             sampler=sampler,
                                             num_workers=0 if dataset.device == 'cuda' else os.cpu_count(),
                                             collate_fn=lambda x: x[0],
                                             pin_memory=(dataset.device == 'cpu'))
    
    train([data_train, None, None], model, args)

    # print final estimation.
    print('intercept')
    print(model.coef_dict['intercept'].coef)
    print('cost, freq, ovt')
    print(model.coef_dict['i'].coef)
    print('income')
    print(model.coef_dict['t'].coef)
    print('ivt')
    print(model.coef_dict['it'].coef)


if __name__ == '__main__':
    main()

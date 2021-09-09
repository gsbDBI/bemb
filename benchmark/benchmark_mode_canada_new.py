"""
Benchmark using the ModeCanada dataset in R.
"""
import argparse
import os
import pdb
import sys
from typing import Optional
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
import yaml


from deepchoice.data import utils
from deepchoice.data import ChoiceDataset


def __null__():
    print('Building the model...')
    model = ConditionalLogitModel(num_items=4,
                                  num_users=1,
                                  coef_variation_dict=args.coef_variation_dict,
                                  num_param_dict=args.num_param_dict)

    model = model.to(args.device)
    print(f'Number of parameters: {model.num_params}')
    
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
                                             num_workers=0,  # 0 if dataset.device == 'cuda' else os.cpu_count(),
                                             collate_fn=lambda x: x[0],
                                             pin_memory=(dataset.device == 'cpu'))
    start = datetime.now()
    train([data_train, None, None], model, args)

    # print final estimation.
    print('intercept')
    print(model.coef_dict['intercept'].coef)
    print('cost, freq, ovt')
    print(model.coef_dict['price_cost'].coef)
    print('income')
    print(model.coef_dict['session_income'].coef)
    print('ivt')
    print(model.coef_dict['price_ivt'].coef)

    print('=' * 10 + 'computing Hessian' + '=' * 10)
    std_dict = model.compute_std(x_dict=dataset.x_dict,
                                 availability=dataset.item_availability,
                                 user_onehot=dataset.user_onehot,
                                 y=dataset.label)

    for var_type in std_dict.keys():
        print(f'Variable Type: {var_type}')
        for i, s in enumerate(std_dict[var_type]):
            c = model.coef_dict[var_type].coef.view(-1,)[i]
            print(f'{c.item():.4f}, {s.item():.4f}')

    print(f'Total time taken: {datetime.now() - start}')


if __name__ == '__main__':
    main()

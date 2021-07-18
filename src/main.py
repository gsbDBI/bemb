"""
This is a draft script.
"""
import argparse
import copy
import os
import pdb
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.data_formatter import stata_to_tensors
# sys.path.extend('./')
from data.dataloader_logit import LogitDataset
from model.conditional_logit_model import ConditionalLogitModel


def main(modifier: Optional[dict]=None):
    arg_path = sys.argv[1]
    # arg_path = './args.yaml'
    with open(arg_path) as file:
        args = argparse.Namespace(**yaml.load(file, Loader=yaml.FullLoader))  # unwrap dictionary loaded.
    
    if modifier is not None:
        for k, v in modifier.items():
            setattr(args, k, v)

    # %%
    # data.
    project_path = '/home/tianyudu/Development/deepchoice'
    travel = pd.read_csv(os.path.join(project_path, 'data/stata/travel.csv'))
    # u: user, i: item, t: time/session.
    var_cols_dict = {
        'u': None,
        'i': None,
        'ui': None,
        't': ['income', 'partysize'],
        'ut': None,
        'it': ['termtime', 'invehiclecost', 'traveltime', 'travelcost'],
        'uit': None
    }
    
    X, user_onehot, A, Y, C = stata_to_tensors(df=travel,
                                               var_cols_dict=var_cols_dict,
                                               session_id='id',
                                               item_id='mode',
                                               choice='choice',
                                               category_id=None,
                                               user_id=None)

    writer = SummaryWriter(log_dir=f'./out/lr={args.learning_rate}')

    # %%
    # create the model.
    model = ConditionalLogitModel(num_items=args.num_items,
                                  num_users=args.num_users,
                                  var_variation_dict=args.var_variation_dict,
                                  var_num_params_dict=args.var_num_params_dict)
    model = model.to(args.device)
    user_onehot, A, Y, C = map(lambda x: x.to(args.device), (user_onehot, A, Y, C))
    for k, v in X.items():
        if v is not None:
            X[k] = v.to(args.device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    last_model = None
    norm_change = 1E10
    epoch = 0
    # run till convergence.
    while (norm_change > 1E-5) and (epoch <= args.num_epochs):
        epoch += 1
        # record metrics across mini-batches.
        epoch_loss, epoch_acc = list(), list()
        
        for i in range(len(Y) // args.batch_size):
            first, last = i * args.batch_size, min(len(Y), (i + 1) * args.batch_size)
            index = torch.arange(first, last)
            
            X_batch = dict()
            for k, x in X.items():
                if isinstance(x, torch.Tensor):
                    X_batch[k] = x[index, :, :]
                else:
                    X_batch[k] = None
            
            y_pred = model(X_batch, A[index, :], user_onehot[index])
            loss = loss_fn(y_pred, Y[index])

            epoch_loss.append(loss.item())
            epoch_acc.append((y_pred.argmax(dim=1) == Y[index]).float().mean().item())

            loss.backward()
            optimizer.step()
        
        # record metrics within at current epoch.
        writer.add_scalar('Loss/Train', np.mean(epoch_loss), epoch)
        writer.add_scalar('Acc/Train', np.mean(epoch_acc), epoch)
       
        # track the status of convergence.
        if last_model is None:
            norm_change = 1E10
        else:
            total_norm = 0.0
            for p_old, p_new in zip(last_model.parameters(), model.parameters()):
                param_norm = (p_old - p_new).data.norm(2)
                total_norm += param_norm.item() ** 2
            norm_change = total_norm ** (1. / 2)
        last_model = copy.deepcopy(model)
        writer.add_scalar('NormChange/Train', norm_change, epoch)

        
if __name__ == '__main__':
    # for lr in tqdm(range(1, 15)):
    #     main({'learning_rate': 10**(-lr)})
    #     main({'learning_rate': 30**(-lr)})
    
    main()
    # for _ in range(10):
        # main({'learning_rate': 1E-7})

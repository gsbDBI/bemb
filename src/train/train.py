"""
The basic training loop.
"""
import argparse
import copy
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

import train.utils as utils


def train(datasets: List[dict],
          model: torch.nn.Module,
          args: argparse.Namespace):
    """The training procedure.

    Args:
        datasets (List[dict]): a list of 3 dictionaries, which contains required information for
            model training/validating/testing. Each dictionary should have at least 'X', 'user_onehot',
            'A', 'Y', and 'C' as keys.
        The first dictionary is the training dataset, the second is the validation dataset,
            the last one is the testing dataset. Put None there to disable validation or testing.
        model (torch.nn.Module): a pytorch model.
        args (argparse.Namespace): a collection of args.
    """
    assert len(datasets) == 3
    assert datasets[0] is not None, 'Training dataset is required.'
    data_train, data_val, data_test = datasets
    do_validation = data_val is not None
    do_test = data_test is not None
    
    X, user_onehot, A, Y, C = (data_train['X'], data_train['user_onehot'],
                               data_train['A'], data_train['Y'], data_train['C'])

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.lr_decay)
    loss_fn = nn.CrossEntropyLoss()

    last_model = None
    param_norm_change = 1E10
    epoch = 0

    # record training progress.
    loss_list, acc_list, ll_list = [], [], []

    if args.batch_size == -1:
        args.batch_size = len(Y)

    # run until convergence in terms of parameter.
    while (param_norm_change > args.param_norm_eps) and (epoch <= args.max_epochs):
        epoch += 1
        # record metrics across batches.
        epoch_loss, epoch_acc = list(), list()
        log_likelihood = 0.0

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

            # get log-likelihood.
            f = nn.Softmax(dim=1)
            prob = f(y_pred).detach()
            log_likelihood += torch.log(prob[torch.arange(len(Y)), Y] + 1e-6).sum()
            # another way of computing it.
            # log_likelihood -= float(loss.item()) * len(index)

            epoch_loss.append(loss.item())
            epoch_acc.append((y_pred.argmax(dim=1) == Y[index]).float().mean().item() * 100)

            loss.backward()
            optimizer.step()
            scheduler.step()
       
        # track the status of convergence.
        if last_model is None:
            param_norm_change = 1E10
        else:
            param_norm_change = utils.diff_in_norm(last_model, model.state_dict())
        last_model = copy.deepcopy(model.state_dict())
 
        loss_list.append(np.mean(epoch_loss))
        acc_list.append(np.mean(epoch_acc))
        ll_list.append(log_likelihood.detach().cpu())
        if epoch % args.eval_epoch == 0:
            print(f'[Epoch={epoch} Train] LL={log_likelihood:5f}, ACC={np.mean(epoch_acc):5f}%, LOSS={np.mean(epoch_loss):5f}, NORM DELTA={param_norm_change:5f}')

    torch.save(model, os.path.join(args.out_dir, 'model.pt'))
    torch.save(prob, os.path.join(args.out_dir, 'prob.pt'))
    print(prob.mean(dim=0))

    # summary plot.
    fig, axes = plt.subplots(ncols=3, figsize=(24, 8))
    axes[0].plot(loss_list, label='loss')
    axes[1].plot(acc_list, label='accuracy')
    axes[2].plot(ll_list, label='log-likelihood')
    for ax in axes:
        ax.legend()
    fig.savefig(os.path.join(args.out_dir, 'curve.png'))

"""
A prototypical training procedure for training deep-choice models.
"""
import argparse
import copy
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

import _deepchoice.train.utils as utils


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """A helper function for computing accuracy of prediction.
    Args:
        outputs: float tensor with hshape(N, num_classes)
        labels: long tensor with shape (N,)
    """
    _, preds = torch.max(outputs, dim=1)
    acc = torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100
    return float(acc)


def train(data_loaders: List[dict],
          model: torch.nn.Module,
          args: argparse.Namespace):
    """The training procedure.

    Args:
        data_loaders (List[dict]): a list of 3 dictionaries, which contains required information for
            model training/validating/testing. Each dictionary should have at least 'X', 'user_onehot',
            'A', 'Y', and 'C' as keys.
        The first dictionary is the training data_loader, the second is the validation data_loader,
            the last one is the testing data_loader. Put None there to disable validation or testing.
        model (torch.nn.Module): a pytorch model.
        args (argparse.Namespace): a collection of args.
    """
    assert len(data_loaders) == 3
    assert data_loaders[0] is not None, 'Training data_loader is required.'
    data_train, data_val, data_test = data_loaders
    do_validation = data_val is not None
    do_test = data_test is not None

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.lr_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5000, gamma=0.1)

    # model's state_dict from last epoch.
    last_model = None
    param_norm_change = 1E10
    epoch = 0

    # record training progress.
    loss_list, acc_list, ll_list = [], [], []

    # run until convergence in terms of parameter.
    # TODO(Tianyu): optionally assess convergence using change in loss.
    while (param_norm_change > args.param_norm_eps) and (epoch <= args.max_epochs):
        epoch += 1
        # running average across mini-batches.
        epoch_loss, epoch_acc = list(), list()
        log_likelihood = 0.0
        model.train()
        for batch in data_train:
            y_pred = model(batch)
            loss = F.cross_entropy(y_pred, batch.label, reduction='mean')
            log_likelihood -= float(loss.item()) * len(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_acc.append(accuracy(outputs=y_pred, labels=batch.label))
        # change learning rate after processing all batches, which should leads to faster convergence.
        # see https://discuss.pytorch.org/t/scheduler-step-after-each-epoch-or-after-each-minibatch/111249
        # for scheduling after each epoch or each mini-batch.
        scheduler.step()
        # assess the status of convergence.
        if last_model is None:
            param_norm_change = 1E10
        else:
            param_norm_change = utils.diff_in_norm(last_model, model.state_dict())
        # record the current model.
        last_model = copy.deepcopy(model.state_dict())

        # average across mini-batches get training loss for the current epoch.
        loss_list.append(np.mean(epoch_loss))
        acc_list.append(np.mean(epoch_acc))
        ll_list.append(log_likelihood)

        # report training progress
        if epoch % args.eval_epoch == 0:
            out = f'[Epoch={epoch} Train] LL={log_likelihood:6f}, ACC={np.mean(epoch_acc):5f}%, LOSS={np.mean(epoch_loss):5f}, NORM DELTA={param_norm_change:5f}'
            if do_validation:
                perf = eval_step(model, data_val, args)
                out += f"[Val] LL={perf['log_likelihood']:5f}, ACC={perf['accuracy']:5f}%, LOSS={perf['loss']:5f}"
            if do_test:
                perf = eval_step(model, data_test, args)
                out += f"[Test] LL={perf['log_likelihood']:5f}, ACC={perf['accuracy']:5f}%, LOSS={perf['loss']:5f}"
            print(out)


@torch.no_grad()
def eval_step(model, data_loader, args) -> dict:
    """Evaluate the model on either the validation or testing set, returns a dictionary of
    evaluation metrics.
    """
    model.eval()
    log_likelihood = 0.0
    # average across batches, but in most cases, validation and test data_loaders should only have 1 batch.
    loss_list, acc_list = list(), list()
    for batch in data_loader:
        y_pred = model(batch)
        y_batch = batch.label
        loss = F.cross_entropy(y_pred, y_batch, reduction='mean')
        loss_list.append(float(loss.item()))
        log_likelihood -= float(loss.item()) * len(y_batch)
        acc_list.append(accuracy(outputs=y_pred, labels=y_batch))
    loss = np.mean(loss_list)
    acc = np.mean(acc_list)
    return {'log_likelihood': log_likelihood, 'loss': loss, 'accuracy': acc}

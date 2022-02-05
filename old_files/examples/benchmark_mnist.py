"""
Benchmark of comparing our model's performance with logistic regression implemented in pytorch on
MNIST dataset.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
# from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

sys.path.append('../src')
from model.conditional_logit_model import ConditionalLogitModel
from train.train import train

DEVICE = 'cuda'


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class MnistModel(nn.Module):
    def __init__(self, input_size: int = 784, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes, bias=False)

    def forward(self, xb):
        xb = xb.reshape(-1, 784).to(DEVICE)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        out = self(images)                    # Generate predictions

        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['val_loss'], result['val_acc']))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def benchmark_pytorch_logreg():
    path = '/home/tianyudu/Development/deepchoice/data/'

    # Download training dataset
    # dataset = MNIST(root=path, download=True)

    dataset = MNIST(root=path,
                    train=True,
                    transform=transforms.ToTensor())

    train_ds, val_ds = random_split(dataset, [50000, 10000])
    len(train_ds), len(val_ds)

    batch_size = 128

    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)

    model = MnistModel().to(DEVICE)

    history = fit(100, 0.001, model, train_loader, val_loader)

    return history


def benchmark_our_model(args):
    path = '/home/tianyudu/Development/deepchoice/data/'
    dataset = MNIST(root=path,
                    train=True,
                    transform=transforms.ToTensor())

    X_all = dataset.train_data
    X_all = X_all.reshape(-1, 784)
    # (num_obs, num_items, num_feats), all classes share the same feature.
    X_all = X_all.reshape(60000, 1, 784).expand(-1, 10, -1).float().clone().to(args.device)
    # (num_obs,)
    y_all = dataset.train_labels.clone().to(args.device)

    # random train-validation split, 50k-10k train-validation.
    val_idx = np.random.choice(len(dataset), size=(10000), replace=False)
    val_mask = np.zeros(len(dataset))
    val_mask[val_idx] = 1
    val_mask = torch.Tensor(val_mask).bool()

    # all 784 features are assigned to t-variable with item-specific coefficients (see args yaml).
    # consider all sessions are from the same user.
    # all 10 digits are available for prediction in all sessions.
    datasets = [
        # training set.
        {'X': {'t': X_all[~val_mask]},
         'Y': y_all[~val_mask],
         'user_onehot': torch.ones(50000, 1).to(args.device),
         'A': torch.ones(50000, 10).bool().to(args.device),
         'C': None},
        # validation set.
        {'X': {'t': X_all[val_mask]},
         'Y': y_all[val_mask],
         'user_onehot': torch.ones(10000, 1).to(args.device),
         'A': torch.ones(10000, 10).bool().to(args.device),
         'C': None},
        # no test set provided.
        None
    ]

    model = ConditionalLogitModel(num_items=args.num_items,
                                  num_users=args.num_users,
                                  var_variation_dict=args.var_variation_dict,
                                  var_num_params_dict=args.var_num_params_dict)

    model = model.to(args.device)
    train(datasets, model, args)


if __name__ == '__main__':
    arg_path = sys.argv[1]
    # arg_path = './args.yaml'
    with open(arg_path) as file:
        # unwrap dictionary loaded.
        args = argparse.Namespace(**yaml.load(file, Loader=yaml.FullLoader))

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    benchmark_pytorch_logreg()
    benchmark_our_model(args)

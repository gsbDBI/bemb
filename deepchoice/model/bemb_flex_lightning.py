"""
BEMB Flex model adopted to PyTorch Lightning.
"""
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from deepchoice.model import BEMBFlex


class LitBEMBFlex(pl.LightningModule):

    def __init__(self, learning_rate: float = 0.3, num_seeds: int=1, **kwargs):
        # use kwargs to pass parameter to BEMB Torch.
        super().__init__()
        self.model = BEMBFlex(**kwargs)
        self.num_needs = num_seeds
        self.learning_rate = learning_rate

    def __str__(self) -> str:
        return str(self.model)

    def training_step(self, batch, batch_idx):
        # with torch.no_grad():
        #     p_all = self.model(batch, all_items=True, return_logit=True)
        #     p1 = p_all[torch.arange(len(batch)), batch.label]
        #     p2 = self.model(batch, all_items=False, return_logit=True)
        #     print(torch.max(torch.abs(p1 - p2)))
        elbo = self.model.elbo(batch, num_seeds=self.num_needs)
        self.log('train_elbo', elbo)
        loss = - elbo
        return loss

    def evaluation_step(self, batch, batch_idx, name, report_accuracy: bool = False):
        # common codes in the validation and test step.
        # name is either val or test.
        # get the log-likelihood.
        # non-differentiable metrics.
        if report_accuracy:
            pred = self.model(batch)
            performance = self.model.get_within_category_accuracy(pred, batch.label)
        else:
            performance = dict()

        # performance[name + '_log_likelihood'] = pred[torch.arange(len(batch)), batch.label].mean().detach().cpu().item()
        performance[name + '_log_likelihood'] = self.model.forward(batch, all_items=False)
        for key, val in performance.items():
            self.log(name + '_' + key, val, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluation_step(batch, batch_idx, 'val', report_accuracy=False)

    def test_step(self, batch, batch_idx):
        self.evaluation_step(batch, batch_idx, 'test', report_accuracy=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

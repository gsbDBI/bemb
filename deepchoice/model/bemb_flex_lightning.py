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
        elbo = self.model.elbo(batch, num_seeds=self.num_needs)
        self.log('train_elbo', elbo)
        loss = - elbo
        return loss

    def validation_step(self, batch, batch_idx):
        # get the log-likelihood.
        pred = self.model(batch)
        # non-differentiable metrics.
        performance = self.model.get_within_category_accuracy(pred, batch.label)
        for key, val in performance.items():
            self.log('val_' + key, val, prog_bar=(key == 'accuracy'))
        LL = pred[torch.arange(len(batch)), batch.label].mean().detach().cpu().item()
        self.log('val_log_likelihood', LL)

    def test_step(self, batch, batch_idx):
        # --------------------------
        # REPLACE WITH YOUR OWN
        # x, y = batch
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        # self.log('test_loss', loss)
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

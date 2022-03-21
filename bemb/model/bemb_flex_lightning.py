"""
PyTorch lightning wrapper for the BEMB Flex model.
"""
import numpy as np
import torch
import pytorch_lightning as pl
from bemb.model import BEMBFlex
from sklearn import metrics

class LitBEMBFlex(pl.LightningModule):

    def __init__(self, learning_rate: float = 0.3, num_seeds: int=1, **kwargs):
        # use kwargs to pass parameter to BEMB Torch.
        super().__init__()
        self.model = BEMBFlex(**kwargs)
        self.num_needs = num_seeds
        self.learning_rate = learning_rate

    def __str__(self) -> str:
        return str(self.model)

    def forward(self, args, kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        elbo = self.model.elbo(batch, num_seeds=self.num_needs)
        self.log('train_elbo', elbo)
        loss = - elbo
        return loss

    def _get_performance_dict(self, batch):
        if self.model.pred_item:
            log_p = self.model(batch, return_type='log_prob', return_scope='all_items', deterministic=True).cpu().numpy()
            num_classes = log_p.shape[1]
            y_pred = np.argmax(log_p, axis=1)
            y_true = batch.item_index.cpu().numpy()
            performance = {'acc': metrics.accuracy_score(y_true=y_true, y_pred=y_pred),
                           'll': - metrics.log_loss(y_true=y_true, y_pred=np.exp(log_p), labels=np.arange(num_classes))}
        else:
            # making binary station.
            pred = self.model(batch, return_type='utility', return_scope='item_index', deterministic=True)
            y_pred = torch.sigmoid(pred).cpu().numpy()
            y_true = batch.label.cpu().numpy()
            performance = {'acc': metrics.accuracy_score(y_true=y_true, y_pred=(y_pred >= 0.5).astype(int)),
                           'll': - metrics.log_loss(y_true=y_true, y_pred=y_pred, eps=1E-5, labels=[0, 1]),
                        #    'auc': metrics.roc_auc_score(y_true=y_true, y_score=y_pred),
                        #    'f1': metrics.f1_score(y_true=y_true, y_pred=(y_pred >= 0.5).astype(int))
                        }
        return performance

    def validation_step(self, batch, batch_idx):
        # LL = self.model.forward(batch, return_type='log_prob', return_scope='item_index', deterministic=True).mean()
        # self.log('val_log_likelihood', LL, prog_bar=True)
        # pred = self.model(batch)
        # performance = self.model.get_within_category_accuracy(pred, batch.label)

        # utility.

        for key, val in self._get_performance_dict(batch).items():
            self.log('val_' + key, val, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # LL = self.model.forward(batch, return_logit=False, all_items=False).mean()
        # self.log('test_log_likelihood', LL)

        # pred = self.model(batch, return_type='utility', return_scope='item_index', deterministic=True)
        # y_pred = torch.sigmoid(pred).cpu().numpy()
        # y_true = batch.label.cpu().numpy()
        # performance = {'acc': metrics.accuracy_score(y_true=y_true, y_pred=(y_pred >= 0.5).astype(int)),
        #                'll': - metrics.log_loss(y_true=y_true, y_pred=y_pred, eps=1E-5, labels=[0, 1]),
        #             #    'auc': metrics.roc_auc_score(y_true=y_true, y_score=y_pred),
        #             #    'f1': metrics.f1_score(y_true=y_true, y_pred=(y_pred >= 0.5).astype(int))
        #                }

        # pred = self.model(batch)
        # performance = self.model.get_within_category_accuracy(pred, batch.label)
        for key, val in self._get_performance_dict(batch).items():
            self.log('test_' + key, val, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

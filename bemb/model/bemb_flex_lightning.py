"""
PyTorch lightning wrapper for the BEMB Flex model, allows for more smooth model training and inference. You can still
use this package without using LitBEMBFlex.

Author: Tianyu Du
Update: Apr. 29, 2022
"""
import time
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn import metrics
from torch_choice.data import ChoiceDataset
from torch_choice.data.utils import create_data_loader

from bemb.model import BEMBFlex


class LitBEMBFlex(pl.LightningModule):

    def __init__(self, learning_rate: float = 0.3, num_seeds: int = 1, **kwargs):
        """The initialization method of the wrapper model.

        Args:
            learning_rate (float, optional): the learning rate of optimization. Defaults to 0.3.
            num_seeds (int, optional): number of random seeds for the Monte Carlo estimation in the variational inference.
                Defaults to 1.
            **kwargs: all keyword arguments used for constructing the wrapped BEMB model.
        """
        # use kwargs to pass parameter to BEMB Torch.
        super().__init__()
        self.model = BEMBFlex(**kwargs)
        self.num_needs = num_seeds
        self.learning_rate = learning_rate

    def __str__(self) -> str:
        return str(self.model)

    def forward(self, args, kwargs):
        """Calls the forward method of the wrapped BEMB model, please refer to the documentaton of the BEMB class
            for detailed definitions of the arguments.

        Args:
            args (_type_): arguments passed to the forward method of the wrapped BEMB model.
            kwargs (_type_): keyword arguments passed to the forward method of the wrapped BEMB model.

        Returns:
            _type_: returns whatever the wrapped BEMB model returns.
        """
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        elbo = self.model.elbo(batch, num_seeds=self.num_needs)
        self.log('train_elbo', elbo)
        loss = - elbo
        return loss

    def _get_performance_dict(self, batch):
        if self.model.pred_item:
            log_p = self.model(batch, return_type='log_prob',
                               return_scope='all_items', deterministic=True).cpu().numpy()
            num_classes = log_p.shape[1]
            y_pred = np.argmax(log_p, axis=1)
            y_true = batch.item_index.cpu().numpy()
            performance = {'acc': metrics.accuracy_score(y_true=y_true, y_pred=y_pred),
                           'll': - metrics.log_loss(y_true=y_true, y_pred=np.exp(log_p), labels=np.arange(num_classes))}
        else:
            # making binary station.
            pred = self.model(batch, return_type='utility',
                              return_scope='item_index', deterministic=True)
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
            self.log('val_' + key, val, prog_bar=True, batch_size=len(batch))

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
            self.log('test_' + key, val, prog_bar=True, batch_size=len(batch))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def fit_model(self, dataset_list: List[ChoiceDataset], batch_size: int=-1, num_epochs: int=10, num_workers: int=8, **kwargs) -> "LitBEMBFlex":
        """A standard pipeline of model training and evaluation.

        Args:
            dataset_list (List[ChoiceDataset]): train_dataset, validation_test, and test_dataset in a list of length 3.
            batch_size (int, optional): batch_size for training and evaluation. Defaults to -1, which indicates full-batch training.
            num_epochs (int, optional): number of epochs for training. Defaults to 10.
            **kwargs: additional keyword argument for the pytorch-lightning Trainer.

        Returns:
            LitBEMBFlex: the trained bemb model.
        """

        def section_print(input_text):
            """Helper function for printing"""
            print('=' * 20, input_text, '=' * 20)
        # present a summary of the model received.
        section_print('model received')
        print(self)

        # present a summary of datasets received.
        section_print('data set received')
        print('[Training dataset]', dataset_list[0])
        print('[Validation dataset]', dataset_list[1])
        print('[Testing dataset]', dataset_list[2])

        # create pytorch dataloader objects.
        train = create_data_loader(dataset_list[0], batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation = create_data_loader(dataset_list[1], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # WARNING: the test step takes extensive memory cost since it computes likelihood for all items.
        # we run the test step with a much smaller batch_size.
        test = create_data_loader(dataset_list[2], batch_size=batch_size // 10, shuffle=False, num_workers=num_workers)

        section_print('train the model')
        trainer = pl.Trainer(gpus=1 if ('cuda' in str(self)) else 0,  # use GPU if the model is currently on the GPU.
                            max_epochs=num_epochs,
                            check_val_every_n_epoch=1,
                            log_every_n_steps=1,
                            **kwargs)
        start_time = time.time()
        trainer.fit(self, train_dataloaders=train, val_dataloaders=validation)
        print(f'time taken: {time.time() - start_time}')

        section_print('test performance')
        trainer.test(self, dataloaders=test)
        return self

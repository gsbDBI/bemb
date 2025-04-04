"""
BEMB Flex model adopted to PyTorch Lightning.
"""
import os

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional import accuracy
from torchmetrics.functional import accuracy
from bemb.model import BEMBFlex
from bemb.model.bemb import parse_utility
from torch_choice.data.utils import create_data_loader


class WeekTrendPreprocessor(nn.Module):
    def __init__(self, num_weeks: int, latent_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_weeks, latent_dim)

    def forward(self, batch):
        # batch.session_obs_w expected, (num_session, 1)
        # convert to batch.session_delta, (num_session, num_latent)
        # session_delta will be considered as a session-specific observable by BEMB.
        batch.session_week = self.emb(batch.session_week_id.long())
        return batch

class LitBEMBFlex(pl.LightningModule):

    def __init__(self, configs=None, user_encoder=None, item_encoder=None, session_encoder=None, category_encoder=None, obs_dict=None, print_info=None, train_data=None, validation_data=None, batch_size: int = -1, num_workers: int = 8, learning_rate: float = 0.3, num_seeds: int=1, num_weeks=0, num_week_trend_latents=0, test_data=None, preprocess=True, lr_decay_type='multi_step_lr', lr_milestones=[], lr_decay=1.0, check_val_every_n_epoch=5, **kwargs):
        # use kwargs to pass parameter to BEMB Torch.
        super().__init__()
        # import pdb; pdb.pdb.set_trace()
        self.save_hyperparameters(ignore=['train_data', 'validation_data', 'test_data'])
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        bemb_params = {key : self.hparams[key] for key in kwargs.keys()}
        self.model = BEMBFlex(**bemb_params)
        self.configs = self.hparams.configs
        # self.configs = configs
        self.num_seeds = self.hparams.num_seeds
        self.batch_size = self.hparams.batch_size
        self.num_workers = self.hparams.num_workers
        self.learning_rate = self.hparams.learning_rate
        self.preprocess =  self.hparams.preprocess
        self.lr_decay_type = self.hparams.lr_decay_type
        self.lr_milestones = self.hparams.lr_milestones
        self.lr_decay = self.hparams.lr_decay
        self.check_val_every_n_epoch = self.hparams.check_val_every_n_epoch
        if self.preprocess:
            self.batch_preprocess = WeekTrendPreprocessor(num_weeks=self.hparams.num_weeks, latent_dim=self.hparams.num_week_trend_latents)

    def __str__(self) -> str:
        return str(self.model)

    def training_step(self, batch, batch_idx):
        if self.preprocess:
            batch = self.batch_preprocess(batch)
        elbo = self.model.elbo(batch, num_seeds=self.num_seeds)
        self.log('train_elbo', elbo)
        loss = - elbo
        return loss

    def validation_step(self, batch, batch_idx):
        if self.preprocess:
            batch = self.batch_preprocess(batch)
        LL = self.model.forward(batch, return_type='log_prob', return_scope='item_index').mean()
        # self.log('val_log_likelihood', LL, prog_bar=True)
        return {'LL':LL}

    # def training_epoch_end(self, outputs):
        # sch = self.lr_schedulers()
        # print("Learning Rate %f" % sch.get_lr)

    def validation_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        # print("Learning Rate %r" % sch.get_lr())
        avg_LL = torch.stack([x["LL"] for x in outputs]).mean()
        # self.log("lr", sch.get_last_lr()[0], prog_bar=True)
        self.log("lr", self.hparams.learning_rate, prog_bar=True)
        self.log("val_log_likelihood", avg_LL, prog_bar=True)

    # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_log_likelihood"])
        # avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        # self.log("ptl/val_loss", avg_loss)
        # self.log("ptl/val_accuracy", avg_acc)

    # def test_step(self, batch, batch_idx):
    #     LL = self.model.forward(batch, return_logit=False, all_items=False).mean()
    #     self.log('test_log_likelihood', LL)

    #     pred = self.model(batch)
    #     performance = self.model.get_within_category_accuracy(pred, batch.label)
    #     for key, val in performance.items():
            # self.log('test_' + key, val, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # return optimizer
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400], gamma=0.99)
        if self.lr_decay_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=self.lr_decay)
            lr_scheduler = {
                'scheduler': scheduler,
                'name': 'lr_log',
                'monitor' : 'val_log_likelihood',
                'frequency' : self.check_val_every_n_epoch,
            }
        elif self.lr_decay_type == 'multi_step_lr':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=self.lr_decay)
            lr_scheduler = scheduler
        else:
            raise ValueError

        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.00, total_steps=200)# steps_per_epoch=len(data_loader), epochs=10)
        # lr_scheduler = scheduler
        return [optimizer], [lr_scheduler]
        # return optimizer

    def train_dataloader(self):
        train_dataloader = create_data_loader(self.train_data,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              num_workers=self.num_workers)
        return train_dataloader

    def val_dataloader(self):
        validation_dataloader = create_data_loader(self.validation_data,
                                                   # batch_size=-1,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.num_workers)
        return validation_dataloader

    def test_dataloader(self):
        if self.test_data is None:
            return None
        test_dataloader = create_data_loader(self.test_data,
                                            # batch_size=-1,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            num_workers=self.num_workers)
        return test_dataloader

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, ):
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        optimizer.step(closure=optimizer_closure)
        with torch.no_grad():
            self.model.clamp_coefs()


    def write_bemb_cpp_format(self):
        model = self.model
        configs = self.hparams.configs
        print_info = self.hparams.print_info
        users = self.hparams.user_encoder.classes_
        items = self.hparams.item_encoder.classes_
        categories = self.hparams.category_encoder.classes_
        sessions = self.hparams.session_encoder.classes_
        user_df = pd.DataFrame(users, columns=['user_id'])
        item_df = pd.DataFrame(items, columns=['item_id'])
        category_df = pd.DataFrame(categories, columns=['category_id'])
        formula = parse_utility(model.utility_formula)
        variations = {'user', 'item', 'category', 'constant', 'session'}
        variations_dfs = {
            'user' : user_df,
            'item' : item_df,
            'category' : category_df,
        }
        variations_ids = {
            'user' : 'user_id',
            'item' : 'item_id',
            'category' : 'category_id',
        }
        for additive_term in formula:
            for coef_name in additive_term['coefficient']:
                variation = coef_name.split('_')[-1]
                params_mean = model.coef_dict[coef_name].variational_mean
                params_std = torch.exp(model.coef_dict[coef_name].variational_logstd)
                for params, moment in zip((params_mean, params_std), ('mean', 'std')):
                    params_df = pd.DataFrame(params.detach().cpu().numpy())
                    if variation != 'constant':
                        params_df = pd.concat((variations_dfs[variation], params_df), axis=1)
                    params_df.to_csv('%s/param_%s_%s.tsv' % (configs.out_dir, coef_name, moment), sep='\t')

        if 'lambda_item' in model.coef_dict:
            print_info['store_dummies'].to_csv('%s/store_dummies.tsv' % (configs.out_dir), sep='\t')
        if 'delta_item' in model.coef_dict:
            print_info['store_weekday_dummies'].to_csv('%s/store_weekday_dummies.tsv' % (configs.out_dir), sep='\t')
        if 'mu_item' in model.coef_dict:
            week_trend_latents = self.batch_preprocess.emb.weight
            week_trend_latents_df = pd.DataFrame(week_trend_latents.detach().cpu().numpy())
            week_trend_latents_df.to_csv('%s/param_weektrends_mean.tsv' % (configs.out_dir), sep='\t')

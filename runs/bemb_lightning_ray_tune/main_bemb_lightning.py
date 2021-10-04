import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import yaml
from deepchoice.data import ChoiceDataset
from deepchoice.data.utils import create_data_loader
from deepchoice.model.bemb_flex_lightning import LitBEMBFlex
from sklearn.preprocessing import LabelEncoder
from termcolor import cprint

from deepchoice.model import BEMBFlex


def load_configs(yaml_file: str):
    with open(yaml_file, 'r') as file:
        data_loaded = yaml.safe_load(file)
    # Add defaults
    defaults = {
        'num_verify_val': 10,
        'early_stopping': {'validation_llh_flat': -1},
        'write_best_model': True
    }
    defaults.update(data_loaded)
    configs = argparse.Namespace(**defaults)
    return configs


def is_sorted(x):
    return all(x == np.sort(x))


def load_tsv(file_name, data_dir):
    return pd.read_csv(os.path.join(data_dir, file_name),
                       sep='\t',
                       index_col=None,
                       names=['user_id', 'item_id', 'session_id', 'quantity'])


configs = load_configs(sys.argv[1])
# ==============================================================================================
# Load standard BEMB inputs.
# ==============================================================================================
train = load_tsv('train.tsv', configs.data_dir)
# read standard BEMB input files.
validation = load_tsv('validation.tsv', configs.data_dir)
test = load_tsv('test.tsv', configs.data_dir)

# ==============================================================================================
# Encode users and items to {0, 1, ..., num-1}.
# ==============================================================================================
# combine data for encoding.
data_all = pd.concat([train, validation, test], axis=0)
# encode user.
user_encoder = LabelEncoder().fit(data_all['user_id'].values)
configs.num_users = len(user_encoder.classes_)
assert is_sorted(user_encoder.classes_)
# encode items.
item_encoder = LabelEncoder().fit(data_all['item_id'].values)
configs.num_items = len(item_encoder.classes_)
assert is_sorted(item_encoder.classes_)

# ==============================================================================================
# user observables
# ==============================================================================================
user_obs = pd.read_csv(os.path.join(configs.data_dir, 'obsUser.tsv'),
                       sep='\t',
                       index_col=0,
                       header=None)
# TODO(Tianyu): there could be duplicate information for each user.
# do we need to catch it in some check process?
user_obs = user_obs.groupby(user_obs.index).first().sort_index()
user_obs = torch.Tensor(user_obs.values)
configs.num_user_obs = user_obs.shape[1]
configs.coef_dim_dict['obsuser_item'] = configs.num_user_obs

# ==============================================================================================
# item observables
# ==============================================================================================
item_obs = pd.read_csv(os.path.join(configs.data_dir, 'obsItem.tsv'),
                       sep='\t',
                       index_col=0,
                       header=None)
item_obs = item_obs.groupby(item_obs.index).first().sort_index()
item_obs = torch.Tensor(item_obs.values)
configs.num_item_obs = item_obs.shape[1]
configs.coef_dim_dict['obsitem_user'] = configs.num_item_obs

# ==============================================================================================
# item availability
# ==============================================================================================
# parse item availability.
# Try and catch? Optionally specify full availability?
a_tsv = pd.read_csv(os.path.join(configs.data_dir, 'availabilityList.tsv'),
                    sep='\t',
                    index_col=None,
                    header=None,
                    names=['session_id', 'item_id'])

# availability ties session as well.
session_encoder = LabelEncoder().fit(a_tsv['session_id'].values)
configs.num_sessions = len(session_encoder.classes_)
assert is_sorted(session_encoder.classes_)
# this loop could be slow, depends on # sessions.
item_availability = torch.zeros(configs.num_sessions, configs.num_items).bool()

a_tsv['item_id'] = item_encoder.transform(a_tsv['item_id'].values)
a_tsv['session_id'] = session_encoder.transform(a_tsv['session_id'].values)

for session_id, df_group in a_tsv.groupby('session_id'):
    # get IDs of items available at this date.
    # this unique is not necessary if the dataset is well-prepared.
    a_item_ids = df_group['item_id'].unique()
    item_availability[session_id, a_item_ids] = True

# ==============================================================================================
# price observables
# ==============================================================================================
df_price = pd.read_csv(os.path.join(configs.data_dir, 'item_sess_price.tsv'),
                       sep='\t',
                       names=['item_id', 'session_id', 'price'])

# only keep prices of relevant items.
mask = df_price['item_id'].isin(item_encoder.classes_)
df_price = df_price[mask]

df_price['item_id'] = item_encoder.transform(df_price['item_id'].values)
df_price['session_id'] = session_encoder.transform(
    df_price['session_id'].values)
df_price = df_price.pivot(index='session_id', columns='item_id')
# NAN prices.
df_price.fillna(0.0, inplace=True)
price_obs = torch.Tensor(df_price.values).view(
    configs.num_sessions, configs.num_items, 1)
configs.num_price_obs = 1

# ==============================================================================================
# create datasets
# ==============================================================================================
dataset_list = list()
for d in (train, validation, test):
    user_index = torch.LongTensor(user_encoder.transform(d['user_id'].values))
    label = torch.LongTensor(item_encoder.transform(d['item_id'].values))
    session_index = torch.LongTensor(
        session_encoder.transform(d['session_id'].values))
    # get the date (aka session_id in the raw dataset) of each row in the dataset, retrieve
    # the item availability information from that date.

    choice_dataset = ChoiceDataset(label=label,
                                   user_index=user_index,
                                   session_index=session_index,
                                   item_availability=item_availability,
                                   user_obs=user_obs,
                                   item_obs=item_obs,
                                   price_obs=price_obs)

    dataset_list.append(choice_dataset)

# ==============================================================================================
# category information
# ==============================================================================================
item_groups = pd.read_csv(os.path.join(configs.data_dir, 'itemGroup.tsv'),
                          sep='\t',
                          index_col=None,
                          names=['item_id', 'category_id'])

# TODO(Tianyu): handle duplicate group information.
item_groups = item_groups.groupby('item_id').first().reset_index()
# filter out items never purchased.
mask = item_groups['item_id'].isin(item_encoder.classes_)
item_groups = item_groups[mask].reset_index(drop=True)
item_groups = item_groups.sort_values(by='item_id')

category_encoder = LabelEncoder().fit(item_groups['category_id'])
configs.num_categories = len(category_encoder.classes_)

# encode them to consecutive integers {0, ..., num_items-1}.
item_groups['item_id'] = item_encoder.transform(
    item_groups['item_id'].values)
item_groups['category_id'] = category_encoder.transform(
    item_groups['category_id'].values)

print('Category sizes:')
print(item_groups.groupby('category_id').size().describe())
item_groups = item_groups.groupby('category_id')['item_id'].apply(list)
category_to_item = dict(zip(item_groups.index, item_groups.values))


# ==================================================================================================
# Pytorch lightning wrapper of BEMB.
# ==================================================================================================

class LitBEMBFlex(pl.LightningModule):
    def __init__(self, hparams, **kwargs):
        # use kwargs to pass parameter to BEMB Torch.
        super().__init__()
        self.model = BEMBFlex(**kwargs)
        self.num_needs = hparams['num_seeds']
        self.learning_rate = hparams['learning_rate']
        self.batch_size = hparams['batch_size']

    def __str__(self) -> str:
        return str(self.model)

    def training_step(self, batch, batch_idx):
        elbo = self.model.elbo(batch, num_seeds=self.num_needs)
        self.log('train_elbo', elbo)
        loss = - elbo
        return loss

    def validation_step(self, batch, batch_idx):
        LL = self.model.forward(
            batch, return_logit=False, all_items=False).mean()
        self.log('val_log_likelihood', LL, prog_bar=True)

    def test_step(self, batch, batch_idx):
        LL = self.model.forward(
            batch, return_logit=False, all_items=False).mean()
        self.log('test_log_likelihood', LL)

        pred = self.model(batch)
        performance = self.model.get_within_category_accuracy(
            pred, batch.label)
        for key, val in performance.items():
            self.log('test_' + key, val, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return create_data_loader(dataset_list[0], batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return create_data_loader(dataset_list[1], batch_size=self.batch_size, shuffle=True, num_workers=8)

    def test_dataloader(self):
        # use smaller batch size for test, which takes more memory.
        return create_data_loader(dataset_list[2], batch_size=10000, shuffle=False, num_workers=8)


if __name__ == '__main__':
    num_samples = 10
    num_epochs = 10

    callback = TuneReportCallback({'val_log_likelihood': 'val_log_likelihood'}, on='validation_end')

    def train_tune(hparams, epochs=10, gpus=1):
        model = LitBEMBFlex(hparams,
                            utility_formula=hparams['utility_formula'],
                            num_users=configs.num_users,
                            num_items=configs.num_items,
                            num_sessions=configs.num_sessions,
                            obs2prior_dict=configs.obs2prior_dict,
                            coef_dim_dict=configs.coef_dim_dict,
                            trace_log_q=configs.trace_log_q,
                            category_to_item=category_to_item,
                            num_user_obs=configs.num_user_obs,
                            num_item_obs=configs.num_item_obs,
                            num_price_obs=configs.num_price_obs)
        trainer = pl.Trainer(
            max_epochs=epochs,
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            gpus=gpus,
            progress_bar_refresh_rate=0,
            logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name='', version='.'),
            callbacks=[callback])
        trainer.fit(model)

    config = {
        'learning_rate': tune.choice([0.01, 0.03, 0.1]),
        'num_seeds': tune.choice([1, 2, 4]),
        'batch_size': tune.choice([100000, -1]),
        'utility_formula': tune.choice([
            'lambda_item',
            'lambda_item + theta_user * alpha_item',
            # 'lambda_item + theta_user * alpha_item + zeta_user * item_obs',
            'lambda_item + theta_user * alpha_item + gamma_user * beta_item * price_obs'
        ])
    }

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    reporter = CLIReporter(parameter_columns=list(config.keys()),
                           metric_columns=list(callback._metrics.keys()))

    analysis = tune.run(
        tune.with_parameters(train_tune, epochs=num_epochs, gpus=1),
        metric='val_log_likelihood',
        mode='max',
        resources_per_trial={'cpu': 16, 'gpu': 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    print("Best hyperparameters found were: ", analysis.best_config)

"""
Example for running BEMB model on the PSID dataset.
"""
import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from torch_choice.data import ChoiceDataset
from torch_choice.data.utils import create_data_loader
from bemb.model import BEMBFlex
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from termcolor import cprint


np.random.seed(42)

def load_configs(yaml_file: str):
    with open(yaml_file, 'r') as file:
        data_loaded = yaml.safe_load(file)
    configs = argparse.Namespace(**data_loaded)
    return configs

# sys.argv[1] should be the yaml file.
configs = load_configs(sys.argv[1])

# ==================================================================================================
# Loading the dataset.
# ==================================================================================================

data_mover_stayer = pd.read_csv(os.path.join(configs.data_dir, 'psid_long_panel.csv'), index_col=0) \
    .rename(columns={'state_1': 'user_id', 'state_2': 'item_id'})

# format year from 2 digits to 4 digits.
f = lambda y: y + 1900 if y > 50 else y + 2000
data_mover_stayer['Year_1'] = data_mover_stayer['Year_1'].apply(f)
data_mover_stayer['Year_2'] = data_mover_stayer['Year_2'].apply(f)

# cprint('Restrict data to movers only.', 'red')
# data_all = data_mover_stayer[data_mover_stayer['user_id'] != data_mover_stayer['item_id']]
data_all = data_mover_stayer

print(f"{len(data_all)} movers found, year_1 scope is {data_all['Year_1'].unique()} ")
state_obs = pd.read_csv(os.path.join(configs.data_dir, 'state_observables.csv'), index_col=0)
print('Number of state(s):', len(state_obs))
print('Number of raw state observables:', state_obs.shape[1], list(state_obs.columns))

# state obs from O*NET.
onet_raw = pd.read_csv(os.path.join(configs.data_dir, 'occ_obs_onet.csv'))
onet_raw.set_index('occ1990dd', inplace=True)
pca = PCA(n_components=4)
onet_reduced = pca.fit_transform(onet_raw.values)
onet_obs = pd.DataFrame(index=onet_raw.index, data=onet_reduced)
onet_obs.columns = [f'ONET_PCA_{i}' for i in range(onet_obs.shape[-1])]
# merge to state obs.
state_obs = state_obs.reset_index()

# match using the 1990dd code.
state_obs['occ1990dd'] = state_obs['state'].map(lambda x: x.split('_')[0]).astype(int)
state_obs = state_obs.merge(onet_obs, how='left', left_on='occ1990dd', right_on='occ1990dd') \
                     .drop(columns=['occ1990dd'])
state_obs.set_index('state', inplace=True)
state_obs.fillna(state_obs.mean(), inplace=True)
print('Number of state observables after adding ONET:', state_obs.shape[1], list(state_obs.columns))
print(f'{len(data_all)} transitions identified with {len(state_obs)} states in total.')

# ==================================================================================================
# ordinal encoding states.
# ==================================================================================================
state_encoder = LabelEncoder().fit(state_obs.index.values)
configs.num_users = len(state_encoder.classes_)
configs.num_items = len(state_encoder.classes_)
assert all(state_encoder.classes_ == np.sort(state_encoder.classes_))

# ==============================================================================================
# user and item observables (derived from the same state observable).
# ==============================================================================================
# user_obs = state_obs.groupby(state_obs.index).first().sort_index()
user_obs = torch.Tensor(state_obs.values)
configs.num_user_obs = user_obs.shape[1]

item_obs = user_obs.clone()
configs.num_item_obs = item_obs.shape[1]

# ==============================================================================================
# session observables
# NOTE: in this model, each row in the dataset is a session.
# ==============================================================================================
# one hot encode categorical variables.
s1 = torch.Tensor(OneHotEncoder().fit_transform(data_all[['SEX', 'RACE', 'Year_1']]).todense())
# continuous variables, fill NANs with the average.
continuous_variables = data_all[['AGE OF INDIVIDUAL_1', 'INCOME_1','NUMBER OF CHILDREN_1', 'YEARS OF EDUCATION_1']]
s2 = torch.Tensor(torch.Tensor(continuous_variables.fillna(continuous_variables.mean()).values))
# combine them to session level observables.
session_obs = torch.cat([s1, s2], dim=1)
configs.num_session_obs = session_obs.shape[1]
configs.num_sessions = len(data_all)
session_index = torch.arange(configs.num_sessions).long()


# ==============================================================================================
# update the configuration.
# ==============================================================================================
configs.coef_dim_dict['delta_constant'] = configs.num_session_obs

# Additional time effect for occupation X year.
# session_occtime = data_all['OCCUPATION_1'].astype(str) + '+' + data_all['Year_1'].astype(str)
# session_occtime = data_all['INDUSTRY_1'].astype(
#     str) + '+' + data_all['Year_1'].astype(str)
# # session_occtime = data_all['Year_1'].astype(str)
# session_occtime = OneHotEncoder().fit_transform(
#     session_occtime.values.reshape(-1, 1)).todense()
# session_occtime = torch.Tensor(session_occtime)

# configs.coef_dim_dict['kappa_constant'] = session_occtime.shape[1]

# encode the starting year of each row (session) of the dataset.
year_encoder = LabelEncoder().fit(data_all['Year_1'].values)
session_year = torch.LongTensor(year_encoder.transform(data_all['Year_1'].values))
configs.num_years = len(year_encoder.classes_)
# ==============================================================================================
# item availability, with shape (num_sessions, num_items/num_states)
# ==============================================================================================
# convert to integer values.
USE_AVAILABILITY = False

if USE_AVAILABILITY:
    data_all['user_id_encoded'] = state_encoder.transform(data_all['user_id'].values)
    data_all['item_id_encoded'] = state_encoder.transform(data_all['item_id'].values)

    item_availability_dict = dict()
    for year in data_all['Year_1'].unique():
        data_before = data_all[data_all['Year_1'] < year]
        if len(data_before) == 0:
            # for the first year.
            item_availability_dict[year] = torch.ones(configs.num_items)
        else:
            # mask out infrequent items.
            item_availability_dict[year] = torch.zeros(configs.num_items)
            item_frequency = data_before['item_id_encoded'].value_counts()
            available_item_list = list(item_frequency[item_frequency >= 1].index)
            item_availability_dict[year][available_item_list] = 1

            if item_availability_dict[year].sum() == 0:
                item_availability_dict[year] = torch.ones(configs.num_items)

    # create the item_availability tensor.
    item_availability = torch.zeros(configs.num_sessions, configs.num_items)
    for year, item_availability_year in item_availability_dict.items():
        item_availability[session_year == int(year_encoder.transform([year])), :] = item_availability_year
else:
    # assume everything to be available.
    item_availability = torch.ones(configs.num_sessions, configs.num_items).bool()
# ==============================================================================================
# create datasets, split into train, validation and test sets with ratio 0.8:0.15:0.15.
# ==============================================================================================
user_index = torch.LongTensor(state_encoder.transform(data_all['user_id'].values))
label = torch.LongTensor(state_encoder.transform(data_all['item_id'].values))

dataset_mover_stayer = ChoiceDataset(label=label,
                                     user_index=user_index,
                                     session_index=session_index,
                                     item_availability=item_availability,
                                     user_obs=user_obs,
                                     item_obs=item_obs,
                                     session_obs=session_obs,
                                     session_year=session_year)

print('Number of MOVER+STAYER observations:', len(dataset_mover_stayer))

dataset_mover = dataset_mover_stayer[dataset_mover_stayer.label != dataset_mover_stayer.user_index]
print('Number of MOVER observations:', len(dataset_mover))

# shuffle the dataset.
N = len(dataset_mover)
shuffle_index = np.random.permutation(N)

train_index = shuffle_index[:int(0.7 * N)]
validation_index = shuffle_index[int(0.7 * N): int(0.85 * N)]
test_index = shuffle_index[int(0.85 * N):]

# splits of dataset.
dataset_mover_splits = [dataset_mover[train_index], dataset_mover[validation_index], dataset_mover[test_index]]

# ==================================================================================================
# Pytorch lightning wrapper of BEMB.
# ==================================================================================================
class YearTrendPreprocessor(nn.Module):
    def __init__(self, num_years: int, latent_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_years, latent_dim)

    def forward(self, batch):
        # batch.session_obs_w expected, (num_session, 1)
        # convert to batch.session_delta, (num_session, num_latent)
        # session_delta will be considered as a session-specific observable by BEMB.
        batch.session_year_emb = self.emb(batch.session_year)
        return batch


class LitBEMBFlex(pl.LightningModule):
    def __init__(self, hparams, train_dataset, val_dataset, test_dataset, **kwargs):
        # use kwargs to pass parameter to BEMB Torch.
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        if 'session_year_emb' in hparams['utility_formula']:
            self.year_embedding = YearTrendPreprocessor(configs.num_years, configs.coef_dim_dict['kappa_constant'])
        else:
            self.year_embedding = None

        if 'K' in hparams.keys():
            kwargs['coef_dim_dict']['theta_user'] = hparams['K']
            kwargs['coef_dim_dict']['beta_item'] = hparams['K']

        self.model = BEMBFlex(**kwargs)
        self.num_needs = hparams['num_seeds']
        self.learning_rate = hparams['learning_rate']
        self.batch_size = hparams['batch_size']

    def __str__(self) -> str:
        return str(self.model)

    def training_step(self, batch, batch_idx):
        if self.year_embedding is not None:
            batch = self.year_embedding(batch)
        LL = self.model.forward(batch, return_logit=False, all_items=False).mean()
        self.log('train_LL', LL, prog_bar=True)

        elbo = self.model.elbo(batch, num_seeds=self.num_needs)
        self.log('train_elbo', elbo)
        loss = - elbo
        return loss

    def validation_step(self, batch, batch_idx):
        if self.year_embedding is not None:
            batch = self.year_embedding(batch)
        LL = self.model.forward(batch, return_logit=False, all_items=False).mean()
        self.log('val_LL', LL, prog_bar=True)

        pred = self.model(batch)
        performance = self.model.get_within_category_accuracy(
            pred, batch.label)
        for key, val in performance.items():
            self.log('val_' + key, val, prog_bar=True)

        self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        if self.year_embedding is not None:
            batch = self.year_embedding(batch)
        LL = self.model.forward(
            batch, return_logit=False, all_items=False).mean()
        self.log('test_LL', LL)

        pred = self.model(batch)
        performance = self.model.get_within_category_accuracy(
            pred, batch.label)
        for key, val in performance.items():
            self.log('test_' + key, val, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return create_data_loader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return create_data_loader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def test_dataloader(self):
        # use smaller batch size for test, which takes more memory.
        return create_data_loader(self.test_dataset, batch_size=10000, shuffle=False, num_workers=8)


def train_model():
    # ==============================================================================================
    # Training
    # ==============================================================================================
    hparams = {'utility_formula': configs.utility,
               'learning_rate': configs.learning_rate,
               'num_seeds': configs.num_mc_seeds,
               'batch_size': configs.batch_size}

    bemb = LitBEMBFlex(
        hparams,
        # datasets
        train_dataset = dataset_mover_splits[0],
        val_dataset = dataset_mover_splits[1],
        test_dataset=dataset_mover_splits[2],
        # model args, will be passed to BEMB constructor.
        prior_variance=configs.prior_variance,
        utility_formula=configs.utility,
        num_users=configs.num_users,
        num_items=configs.num_items,
        num_sessions=configs.num_sessions,
        obs2prior_dict=configs.obs2prior_dict,
        coef_dim_dict=configs.coef_dim_dict,
        trace_log_q=configs.trace_log_q,
        num_user_obs=configs.num_user_obs,
        num_item_obs=configs.num_item_obs
    )
    trainer = pl.Trainer(gpus=1,
                         max_epochs=configs.num_epochs,
                         check_val_every_n_epoch=1,
                         log_every_n_steps=1)
                        #  callbacks=[EarlyStopping(monitor='val_LL', patience=5, mode='max')])

    start_time = time.time()
    trainer.fit(bemb)
    trainer.test(bemb)
    cprint(f'time taken: {time.time() - start_time}', 'red')
    # save the created dataset and trained model.
    torch.save(hparams, './hparams.pt')
    torch.save(configs, './configs.pt')
    torch.save(bemb.state_dict(), './state_dict.pt')
    torch.save(dataset_mover_stayer, './dataset_mover_stayer.pt')
    torch.save(dataset_mover, './dataset_mover.pt')
    torch.save(dataset_mover_splits, './dataset_mover_splits.pt')

    return bemb


def tune_model():
    # ==============================================================================================
    # Hyper-parameter Tuning
    # ==============================================================================================
    num_samples = 50
    num_epochs = 200

    callback1 = TuneReportCallback({
        'val_LL': 'val_LL',
        'val_accuracy': 'val_accuracy',
        'val_precision': 'val_precision',
        'val_recall': 'val_recall',
        'val_f1score': 'val_f1score',
        'train_LL': 'train_LL',
        'train_elbo': 'train_elbo',
        'test_LL': 'test_LL',
        'test_accuracy': 'test_accuracy'}, on='validation_end')

    def train_tune(hparams, epochs=10, gpus=1):
        model = LitBEMBFlex(hparams,
                            # datasets
                            train_dataset = dataset_mover_splits[0],
                            val_dataset = dataset_mover_splits[1],
                            test_dataset=dataset_mover_splits[2],
                            prior_variance=hparams['prior_variance'],
                            utility_formula=hparams['utility_formula'],
                            num_users=configs.num_users,
                            num_items=configs.num_items,
                            num_sessions=configs.num_sessions,
                            obs2prior_dict=configs.obs2prior_dict,
                            coef_dim_dict=configs.coef_dim_dict,
                            trace_log_q=configs.trace_log_q,
                            num_user_obs=configs.num_user_obs,
                            num_item_obs=configs.num_item_obs)

        trainer = pl.Trainer(
            max_epochs=epochs,
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            gpus=gpus,
            progress_bar_refresh_rate=0,
            logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name='', version='.'),
            callbacks=[callback1, EarlyStopping(monitor='val_LL', patience=5, mode='max')])
        trainer.fit(model)
        trainer.test(model)

    config = {
        'learning_rate': tune.choice([0.01, 0.03, 0.1, 0.3]),
        'prior_variance': tune.choice([0.1, 1.0, 3.0, 10, 30, 100]),
        'K': tune.choice([10, 15, 20, 30]),
        'num_seeds': tune.choice([8, 32]),
        'batch_size': tune.choice([1000]),
        'utility_formula': tune.grid_search([
            'lambda_item',
            'theta_user * beta_item',
            'lambda_item + theta_user * beta_item',
            'lambda_item + theta_user * beta_item + delta_item * session_obs',
            'lambda_item + theta_user * beta_item + kappa_constant * session_year_emb',
            'lambda_item + theta_user * beta_item + delta_item * session_obs + kappa_constant * session_year_emb'
        ])
    }

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    reporter = CLIReporter(parameter_columns=list(config.keys()),
                           metric_columns=list(callback1._metrics.keys()))
    analysis = tune.run(
        tune.with_parameters(train_tune, epochs=num_epochs, gpus=1),
        metric='val_LL',
        mode='max',
        resources_per_trial={'cpu': 16, 'gpu': 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    print("Best hyperparameters found were: ", analysis.best_config)
    analysis.results_df.to_csv('./ray_tune_results_' + str(datetime.now()) + '.csv')


def markov_baseline():
    # for reference.
    prob_matrix = torch.zeros(configs.num_users, configs.num_items)
    for i in range(len(dataset_mover_splits[0])):
        user = dataset_mover_splits[0].user_index[i]
        item = dataset_mover_splits[0].label[i]
        prob_matrix[user, item] += 1

    out_deg = prob_matrix.sum(dim=1)
    out_deg[out_deg == 0] = 1
    prob_matrix /= out_deg.view(-1, 1)

    d = dataset_mover_splits[2]  # test set.
    pred = prob_matrix[d.user_index].argmax(dim=1)
    acc = (pred == d.label).float().mean()
    ll = prob_matrix[d.user_index, d.label].mean()
    print(f'accuracy={acc}, LL={torch.log(ll)}')


if __name__ == '__main__':
    if sys.argv[2] == 'train':
        bemb = train_model()
        theta_user = bemb.model.coef_dict['theta_user'].variational_mean.detach().numpy()
        beta_item = bemb.model.coef_dict['beta_item'].variational_mean.detach().numpy()

        pd.DataFrame(theta_user, index=list(state_encoder.classes_)).to_csv('./theta_user.csv')
        pd.DataFrame(beta_item, index=list(state_encoder.classes_)).to_csv('./beta_item.csv')
        # compute the utility.
        # U = bemb.model(dataset_all, return_logit=True).detach().numpy()
        # df_utility = pd.DataFrame(data=U, columns=list(state_encoder.classes_))
        # df_utility['user_id'] = data_mover_stayer['user_id'].values
        # df_utility['item_id'] = data_mover_stayer['item_id'].values
        # df_utility.to_csv('./utility.csv')
    elif sys.argv[2] == 'tune':
        tune_model()

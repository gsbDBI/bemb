import argparse
import os
import sys
import time
from pprint import pprint

import deepchoice
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from deepchoice.data import ChoiceDataset
from deepchoice.data.utils import create_data_loader
from deepchoice.model import BEMB
from sklearn.preprocessing import LabelEncoder
from termcolor import cprint


def load_configs(yaml_file: str):
    with open(yaml_file, 'r') as file:
        data_loaded = yaml.safe_load(file)
    configs = argparse.Namespace(**data_loaded)
    return configs


def load_params_to_model(model, path) -> None:
    def load_cpp_tsv(file):
        df = pd.read_csv(os.path.join(path, file), sep='\t', index_col=0, header=None)
        return torch.Tensor(df.values[:, 1:])

    cpp_theta_mean = load_cpp_tsv('param_theta_mean.tsv')
    cpp_theta_std = load_cpp_tsv('param_theta_std.tsv')

    cpp_alpha_mean = load_cpp_tsv('param_alpha_mean.tsv')
    cpp_alpha_std = load_cpp_tsv('param_alpha_std.tsv')

    # theta user
    model.variational_dict['theta_user'].mean.data = cpp_theta_mean.to(model.device)
    model.variational_dict['theta_user'].logstd.data = torch.log(cpp_theta_std).to(model.device)
    # alpha item
    model.variational_dict['alpha_item'].mean.data = cpp_alpha_mean.to(model.device)
    model.variational_dict['alpha_item'].logstd.data = torch.log(cpp_alpha_std).to(model.device)


def is_sorted(x):
    return all(x == np.sort(x))


def load_tsv(file_name):
    return pd.read_csv(os.path.join(data_dir, file_name),
                       sep='\t',
                       index_col=None,
                       names=['user_id', 'item_id', 'session_id', 'quantity'])


if __name__ == '__main__':
    cprint('Your are running a debugging script!', 'red')
    data_dir = '/home/tianyudu/Data/PSID_V1/all_transitions.csv'

    # sys.argv[1] should be the yaml file.
    configs = load_configs(sys.argv[1])

    # read standard BEMB input files.
    data_all = pd.read_csv(data_dir, index_col=0) \
                 .rename(columns={'state_out': 'user_id', 'state_in': 'item_id'})

    # ordinal encoding users and items.
    user_encoder = LabelEncoder().fit(data_all['user_id'].values)
    configs.num_users = len(user_encoder.classes_)
    assert is_sorted(user_encoder.classes_)

    item_encoder = LabelEncoder().fit(data_all['item_id'].values)
    configs.num_items = len(item_encoder.classes_)
    assert is_sorted(item_encoder.classes_)

    # ==============================================================================================
    # user observables
    # ==============================================================================================
    # user_obs = pd.read_csv(os.path.join(data_dir, 'obsUser.tsv'),
    #                        sep='\t',
    #                        index_col=0,
    #                        header=None)
    # # TODO(Tianyu): there could be duplicate information for each user.
    # # do we need to catch it in some check process?
    # user_obs = user_obs.groupby(user_obs.index).first().sort_index()
    # user_obs = torch.Tensor(user_obs.values)
    # configs.num_user_obs = user_obs.shape[1]
    user_obs = None
    configs.num_user_obs = None

    # # ==============================================================================================
    # # item observables
    # # ==============================================================================================
    # item_obs = pd.read_csv(os.path.join(data_dir, 'obsItem.tsv'),
    #                        sep='\t',
    #                        index_col=0,
    #                        header=None)
    # item_obs = item_obs.groupby(item_obs.index).first().sort_index()
    # item_obs = torch.Tensor(item_obs.values)
    # configs.num_item_obs = item_obs.shape[1]
    item_obs = None
    configs.num_item_obs = None

    # ==============================================================================================
    # item availability
    # ==============================================================================================
    # parse item availability.
    # TODO(Tianyu): not implemented yet ~
    item_availability = None

    # ==============================================================================================
    # create datasets
    # ==============================================================================================
    dataset_list = list()
    # shuffle the dataset.
    data_all = data_all.sample(frac=1)
    N = len(data_all)
    train = data_all.iloc[:int(0.5 * N)]
    validation = data_all.iloc[int(0.5 * N): int(0.833 * N)]
    test = data_all.iloc[int(0.166 * N):]

    for d in (train, validation, test):
        user_idx = user_encoder.transform(d['user_id'].values)

        user_onehot = torch.zeros(len(d), configs.num_users)
        user_onehot[torch.arange(len(d)), user_idx] = 1
        user_onehot = user_onehot.long()

        label = torch.LongTensor(item_encoder.transform(d['item_id'].values))

        choice_dataset = ChoiceDataset(label=label,
                                       user_onehot=user_onehot,
                                       item_availability=item_availability,
                                       user_obs=user_obs,
                                       item_obs=item_obs)

        dataset_list.append(choice_dataset)

    # ==============================================================================================
    # category information
    # ==============================================================================================
    # item_groups = pd.read_csv(os.path.join(data_dir, 'itemGroup.tsv'),
    #                           sep='\t',
    #                           index_col=None,
    #                           names=['item_id', 'category_id'])

    # # TODO(Tianyu): handle duplicate group information.
    # item_groups = item_groups.groupby('item_id').first().reset_index()
    # # filter out items never purchased.
    # mask = item_groups['item_id'].isin(item_encoder.classes_)
    # item_groups = item_groups[mask].reset_index(drop=True)
    # item_groups = item_groups.sort_values(by='item_id')

    # category_encoder = LabelEncoder().fit(item_groups['category_id'])
    # configs.num_categories = len(category_encoder.classes_)

    # # encode them to consecutive integers {0, ..., num_items-1}.
    # item_groups['item_id'] = item_encoder.transform(
    #     item_groups['item_id'].values)
    # item_groups['category_id'] = category_encoder.transform(
    #     item_groups['category_id'].values)

    # print('Category sizes:')
    # print(item_groups.groupby('category_id').size().describe())
    # item_groups = item_groups.groupby('category_id')['item_id'].apply(list)
    # category_to_item = dict(zip(item_groups.index, item_groups.values))
    category_to_item = None

    # ==============================================================================================
    # create data loaders.
    # ==============================================================================================

    dataloaders = dict()
    for dataset, partition in zip(dataset_list, ('train', 'validation', 'test')):
        dataset = dataset.to(configs.device)
        dataloader = create_data_loader(dataset, configs)
        dataloaders[partition] = dataloader

    # ==============================================================================================
    # create model
    # ==============================================================================================

    model = BEMB(num_users=configs.num_users,
                 num_items=configs.num_items,
                 obs2prior_dict=configs.obs2prior_dict,
                 latent_dim=configs.latent_dim,
                 trace_log_q=configs.trace_log_q,
                 category_to_item=category_to_item,
                 num_user_obs=configs.num_user_obs,
                 num_item_obs=configs.num_item_obs,
                 likelihood='all'
                 ).to(configs.device)

    # print(model.variational_dict['theta_user'].mean)
    # # breakpoint()
    # load_params_to_model(model, '/home/tianyudu/Data/MoreSupermarket/t79338-n2123-m1109-k20-users3-lik3-shuffle0-eta0.005-zF0.1-nS10-batch100000-run_K20_1000_rmsprop_bs100000')
    # print(model.variational_dict['theta_user'].mean)

    start_time = time.time()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=1)

    print(80 * '=')
    print(model)
    print(80 * '=')

    # ==============================================================================================
    # training
    # ==============================================================================================

    performance_by_epoch = list()

    for i in range(configs.num_epochs):
        total_loss = torch.scalar_tensor(0.0).to(configs.device)

        for batch in dataloaders['train']:
            # maximize the ELBO.
            loss = - model.elbo(batch, num_seeds=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()

        scheduler.step()
        if i % (configs.num_epochs // 10) == 0:
            # report training progress, report 10 times in total.
            with torch.no_grad():
                performance = {'iteration': i,
                               'duration_seconds': time.time() - start_time}
                # compute performance for each data partition.
                for partition in ('train', 'validation', 'test'):
                    metrics = ['log_likelihood', 'accuracy', 'precision', 'recall', 'f1score']
                    for m in metrics:
                        performance[partition + '_' + m] = list()
                    # compute performance for each batch.
                    for batch in dataloaders[partition]:
                        pred = model(batch)  # (num_sessions, num_items) log-likelihood.
                        LL = pred[torch.arange(len(batch)), batch.label].mean().detach().cpu().item()
                        performance[partition + '_log_likelihood'].append(LL)
                        accuracy_metrics = model.get_within_category_accuracy(pred, batch.label)
                        for key, val in accuracy_metrics.items():
                            performance[partition + '_' + key].append(val)

                for key, val in performance.items():
                    performance[key] = np.mean(val)

                performance_by_epoch.append(performance)
                # pprint(performance)
                print(f"{performance['train_accuracy']=:}, {performance['test_accuracy']=:}")
                print(f"{performance['train_log_likelihood']=:}, {performance['test_log_likelihood']=:}")
                print(f'Epoch [{i}] negative elbo (the lower the better)={total_loss}')
    print(f'Time taken: {time.time() - start_time: 0.1f} seconds.')
    log = pd.DataFrame(performance_by_epoch)

    # ==============================================================================================
    # save results
    # ==============================================================================================

    os.system(f'mkdir {configs.out_dir}')
    log.to_csv(os.path.join(configs.out_dir, 'performance_log_by_epoch.csv'))

    # save model weights
    torch.save(model, os.path.join(configs.out_dir, 'model.pt'))
    torch.save(model.state_dict(), os.path.join(configs.out_dir, 'state_dict.pt'))
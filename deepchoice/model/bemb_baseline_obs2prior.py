import argparse
import yaml
import time
import os
from pprint import pprint

import deepchoice
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from deepchoice.data import ChoiceDataset
from deepchoice.data.utils import create_data_loader
from deepchoice.model import BEMB
from termcolor import cprint

from sklearn.preprocessing import LabelEncoder


# rename command line configs.
arg_old2new = {
    
}


def load_configs(yaml_file: str):
    with open(yaml_file, 'r') as file:
        data_loaded = yaml.safe_load(file)
    breakpoint()
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
    # for debugging.
    data_dir = '/home/tianyudu/Data/MoreSupermarket/tsv'

    # sys.argv[1] should be the yaml file.
    configs = load_configs(sys.argv[1])

    train = load_tsv('train.tsv')
    validation = load_tsv('validation.tsv')
    test = load_tsv('test.tsv')

    data_all = pd.concat([train, validation, test], axis=0)

    # ordinal encoding users and items.
    user_encoder = LabelEncoder().fit(data_all['user_id'].values)
    configs.num_users = len(user_encoder.classes_)
    assert is_sorted(user_encoder.classes_)

    item_encoder = LabelEncoder().fit(data_all['item_id'].values)
    configs.num_items = len(item_encoder.classes_)
    assert is_sorted(item_encoder.classes_)

    # load observables from legacy C++ version.
    user_obs = pd.read_csv(os.path.join(data_dir, 'obsUser.tsv'), sep='\t', index_col=0, header=None)
    # TODO(Tianyu): there could be duplicate information for each user.
    # do we need to catch it in some check process?
    user_obs = user_obs.groupby(user_obs.index).first().sort_index()
    user_obs = torch.Tensor(user_obs.values)
    configs.num_user_obs = user_obs.shape[1]

    item_obs = pd.read_csv(os.path.join(data_dir, 'obsItem.tsv'), sep='\t', index_col=0, header=None)
    item_obs = item_obs.groupby(item_obs.index).first().sort_index()
    item_obs = torch.Tensor(item_obs.values)
    configs.num_item_obs = item_obs.shape[1]

    # parse item availability.
    # TODO.

    # prepare the data loaders.
    dataset_list = list()
    for d in (train, validation, test):
        user_idx = user_encoder.transform(d['user_id'].values)

        user_onehot = torch.zeros(len(d), num_users)
        user_onehot[torch.arange(len(d)), user_idx] = 1
        user_onehot = user_onehot.long()

        label = torch.LongTensor(item_encoder.transform(d['item_id'].values))

        choice_dataset = ChoiceDataset(label=label,
                                       user_onehot=user_onehot,
                                       item_availability=None,  # for now, available items only.
                                       user_obs=user_obs,
                                       item_obs=item_obs)
        dataset_list.append(choice_dataset)

    # load category information.
    item_groups = pd.read_csv(os.path.join(data_dir, 'itemGroup.tsv'),
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

    item_groups['item_id'] = item_encoder.transform(
        item_groups['item_id'].values)
    item_groups['category_id'] = category_encoder.transform(
        item_groups['category_id'].values)

    item_groups = item_groups.groupby('category_id')['item_id'].apply(list)
    category_to_item = dict(zip(item_groups.index, item_groups.values))

    # dataloaders = [create_data_loader(dataset, configs) for dataset in dataset_list]
    # only care the training set for now.
    dataloaders = dict()
    for dataset, partition in zip(dataset_list, ('train', 'validation', 'test')):
        dataset = dataset.to(DEVICE)
        dataloader = create_data_loader(dataset, configs)
        dataloaders[partition] = dataloader

    # create model.
    obs2prior_dict = {
        # 'lambda_item': False,
        'theta_user': False,
        'alpha_item': False,
        # 'zeta_user': False,
        # 'lota_item': False,
        # 'gamma_user': True,
        # 'beta_item': True
    }

    model = BEMB(num_users=configs.num_users,
                 num_items=configs.num_items,
                 obs2prior_dict=configs.obs2prior_dict,
                 latent_dim=configs.latent_dim,
                 trace_log_q=configs.trace_log_q,
                 category_to_item=category_to_item,
                 num_user_obs=configs.num_user_obs,
                 num_item_obs=configs.num_item_obs
                 ).to(DEVICE)

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

    performance_by_epoch = list()

    for i in range(NUM_EPOCHS):
        total_loss = torch.scalar_tensor(0.0).to(DEVICE)

        for batch in dataloaders['train']:
            # maximize the ELBO.
            loss = - model.elbo(batch, num_seeds=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()

        scheduler.step()
        if i % (NUM_EPOCHS // 10) == 0:
            # report training progress, report 100 times in total.
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
                pprint(performance)
                print(f'Epoch [{i}] negative elbo (the lower the better)={total_loss}')
    print(f'Time taken: {time.time() - start_time: 0.1f} seconds.')
    log = pd.DataFrame(performance_by_epoch)
    log.to_csv('./training_log.csv')
    quit()
    # visualize the training performance.

    # same fitted parameters for latter comparison.
    theta = model.variational_dict['theta_user'].mean
    alpha = model.variational_dict['alpha_item'].mean
    torch.save(theta, './output/theta.pt')
    torch.save(alpha, './output/alpha.pt')
    cprint('Done.', 'green')

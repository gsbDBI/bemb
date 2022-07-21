import argparse

import numpy as np
import torch
import yaml
from sklearn.preprocessing import LabelEncoder
from termcolor import cprint
from torch_choice.data import ChoiceDataset
from bemb.model import LitBEMBFlex
from bemb.utils.run_helper import run
import matplotlib.pyplot as plt
import seaborn as sns


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


def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return np.array([x/mag for x in vec])

# configurations.
NUM_ITEMS = 50
NUM_USERS = 100
NUM_SESSIONS = 5000

NUM_USER_TYPES = 20
NUM_ITEM_TYPES = 30


def simulate_dataset():
    # generate simulations.
    # simulation configurations

    # generate types
    user_types =  np.random.randint(low=0, high=NUM_USER_TYPES, size=NUM_USERS)
    item_types =  np.random.randint(low=0, high=NUM_ITEM_TYPES, size=NUM_ITEMS)

    def generate_latents(num_types, num_samples, dim):
        types = np.random.randint(low=0, high=num_types, size=num_samples)
        # mu_list = np.random.rand(num_types, dim) * 50  # scale to [-50, 50]
        mu_list = np.stack([make_rand_vector(dim) for _ in range(num_types)], axis=0) * 10
        # mu_list = np.random.rand(num_types, dim) * 50  # scale to [-50, 50]
        src = np.random.randn(num_samples, dim)  # N(0, I)
        latents = src + mu_list[types, :]
        return mu_list, types, latents

    user_centers, user_types, user_latents = generate_latents(NUM_USER_TYPES, NUM_USERS, 2)
    item_centers, item_types, item_latents = generate_latents(NUM_ITEM_TYPES, NUM_ITEMS, 2)


    plt.close()
    fig, ax = plt.subplots()
    type_list, latent_list, marker = (user_types, user_latents, '*')
    for t in np.unique(type_list):
        ax.scatter(latent_list[type_list == t, 0], latent_list[type_list == t, 1], marker=marker)
    plt.savefig('./user.png')

    plt.close()
    fig, ax = plt.subplots()
    type_list, latent_list, marker = (item_types, item_latents, 'o')
    for t in np.unique(type_list):
        ax.scatter(latent_list[type_list == t, 0], latent_list[type_list == t, 1], marker=marker)
    plt.savefig('./item.png')


    # get choices.
    potential = user_latents @ item_latents.T
    # potential = potential + np.random.gumbel(NUM_USERS, NUM_ITEMS) * potential.mean()
    potential = torch.log_softmax(torch.Tensor(potential), dim=1)
    # potential = potential.numpy()

    out = torch.multinomial(potential, num_samples=NUM_SESSIONS, replacement=True)

    plt.close()
    sns.heatmap(potential)
    plt.savefig('./temp2.png')

    user_index = np.random.randint(low=0, high=NUM_USERS, size=NUM_SESSIONS)
    p = potential[user_index, :]
    label = out[user_index, torch.arange(NUM_SESSIONS)]

    # label = p.argmax(axis=1)
    user_obs = np.zeros((NUM_USERS, NUM_USER_TYPES))
    user_obs[np.arange(NUM_USERS), user_types] = 1
    user_obs = user_obs.astype(int)

    item_obs = np.zeros((NUM_ITEMS, NUM_ITEM_TYPES))
    item_obs[np.arange(NUM_ITEMS), item_types] = 1
    item_obs = item_obs.astype(int)

    choice_dataset = ChoiceDataset(label=torch.LongTensor(label),
                                   user_index=torch.LongTensor(user_index),
                                   session_index=torch.arange(len(label)),
                                   item_availability=None,
                                   user_obs=torch.Tensor(user_obs),
                                   item_obs=torch.Tensor(item_obs))

    return choice_dataset


if __name__ == '__main__':
    cprint('Your are running an example script.', 'green')
    # sys.argv[1] should be the yaml file.
    configs = load_configs('./configs.yaml')

    # choice_dataset = simulate_dataset()
    # # shuffle the dataset.
    # N = len(choice_dataset)
    # shuffle_index = np.random.permutation(N)

    # train_index = shuffle_index[:int(0.7 * N)]
    # validation_index = shuffle_index[int(0.7 * N): int(0.85 * N)]
    # test_index = shuffle_index[int(0.85 * N):]

    # # splits of dataset.
    # dataset_list = [choice_dataset[train_index], choice_dataset[validation_index], choice_dataset[test_index]]
    # torch.save(dataset_list, './dataset_list.pt')
    dataset_list = torch.load('./dataset_list.pt')

    # ==============================================================================================
    # pytorch-lightning training
    # ==============================================================================================
    bemb = LitBEMBFlex(
        # trainings args.
        learning_rate=configs.learning_rate,
        num_seeds=configs.num_mc_seeds,
        # model args, will be passed to BEMB constructor.
        utility_formula=configs.utility,
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        obs2prior_dict=configs.obs2prior_dict,
        coef_dim_dict=configs.coef_dim_dict,
        trace_log_q=configs.trace_log_q,
        num_user_obs=NUM_USER_TYPES,
        num_item_obs=NUM_ITEM_TYPES,
    )

    bemb = bemb.to(configs.device)
    bemb = run(bemb, dataset_list, batch_size=configs.batch_size, num_epochs=configs.num_epochs)

    breakpoint()
    # ==============================================================================================
    # inference example
    # ==============================================================================================
    with torch.no_grad():
        # disable gradient tracking to save computational cost.
        utility_chosen = bemb.model(dataset_list[2], return_logit=True, all_items=False)
        # uses much higher memory!
        utility_all = bemb.model(dataset_list[2], return_logit=True, all_items=True)

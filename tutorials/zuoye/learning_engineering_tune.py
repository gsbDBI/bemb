import argparse
import os

import numpy as np
import torch
from bemb.model import LitBEMBFlex
from bemb.utils.run_helper import run

from load_data import load_data


def main(dataset):
    num_users = len(torch.unique(dataset.user_index))
    num_items = len(torch.unique(dataset.item_index))
    num_item_obs = dataset.item_obs.shape[-1]

    # ----
    # for debugging only, run on a smaller sample.
    # dataset = dataset[np.random.permutation(len(dataset))[:int(0.1*len(dataset))]]
    # ----

    idx = np.random.permutation(len(dataset))
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_idx = idx[:train_size]
    val_idx = idx[train_size: train_size + val_size]
    test_idx = idx[train_size + val_size:]

    dataset_list = [dataset[train_idx], dataset[val_idx], dataset[test_idx]]

    # ==============================================================================================
    # pytorch-lightning training
    # ==============================================================================================
    def compute(hparam):
        bemb = LitBEMBFlex(
            # trainings args.
            learning_rate=0.03,
            pred_item=False,
            num_seeds=1,
            # model args, will be passed to BEMB constructor.
            utility_formula=hparam['utility_formula'],
            num_users=num_users,
            num_items=num_items,
            obs2prior_dict=hparam['obs2prior_dict'],
            coef_dim_dict=hparam['coef_dim_dict'],
            trace_log_q=True,
            num_item_obs=num_item_obs,
            prior_variance=hparam['prior_variance']
        )

        bemb = bemb.to('cuda')
        bemb = run(
            bemb,
            dataset_list,
            batch_size=len(dataset) //
            20,
            num_epochs=50)
        return bemb

    # ==============================================================================================
    # tune different configuration.
    # ==============================================================================================
    hparam_list = list()
    # hparam_list.append({
    #     'obs2prior_dict': {'theta_user': False, 'alpha_item': False},
    #     'coef_dim_dict': {'theta_user': 10, 'alpha_item': 10},
    #     'utility_formula': 'theta_user * alpha_item',
    # })

    # hparam_list.append({
    #     'obs2prior_dict': {'lambda_item': False, 'theta_user': False, 'alpha_item': False},
    #     'coef_dim_dict': {'lambda_item': 1, 'theta_user': 10, 'alpha_item': 10},
    #     'utility_formula': 'lambda_item + theta_user * alpha_item',
    # })

    # hparam_list.append({
    #     'obs2prior_dict': {'lambda_item': True, 'theta_user': False, 'alpha_item': True},
    #     'coef_dim_dict': {'lambda_item': 1, 'theta_user': 10, 'alpha_item': 10},
    #     'utility_formula': 'lambda_item + theta_user * alpha_item',
    # })

    # hparam_list.append({
    #     'obs2prior_dict': {'lambda_item': True, 'theta_user': False, 'alpha_item': True, 'eta_constant': False},
    #     'coef_dim_dict': {'lambda_item': 1, 'theta_user': 10, 'alpha_item': 10, 'eta_constant': num_item_obs},
    #     'utility_formula': 'lambda_item + theta_user * alpha_item + eta_constant * item_obs',
    # })

    # hparam_list.append({
    #     'obs2prior_dict': {'lambda_item': True, 'theta_user': False, 'alpha_item': True, 'eta_user': False},
    #     'coef_dim_dict': {'lambda_item': 1, 'theta_user': 10, 'alpha_item': 10, 'eta_user': num_item_obs},
    #     'utility_formula': 'lambda_item + theta_user * alpha_item + eta_user * item_obs',
    # })

    # for s in [0.03, 0.1, 0.3, 3, 10]:
    #     hparam_list.append({
    #         'obs2prior_dict': {'lambda_item': True, 'theta_user': False, 'alpha_item': True, 'eta_user': False},
    #         'coef_dim_dict': {'lambda_item': 1, 'theta_user': 10, 'alpha_item': 10, 'eta_user': num_item_obs},
    #         'utility_formula': 'lambda_item + theta_user * alpha_item + eta_user * item_obs',
    #         'prior_variance': s,
    #     })

    # hparam_list.append({
    #     'obs2prior_dict': {'lambda_item': True, 'theta_user': False, 'alpha_item': True, 'eta_user': False},
    #     'coef_dim_dict': {'lambda_item': 1, 'theta_user': 5, 'alpha_item': 5, 'eta_user': num_item_obs},
    #     'utility_formula': 'lambda_item + theta_user * alpha_item + eta_user * item_obs',
    #     'prior_variance': 10,
    # })

    # =======
    # for prior_std in [0.001, 0.003, 0.01, 0.03, 0.1, 1, 10]:
    #     for obs2prior in [True, False]:
    #         hparam_list.append({
    #             'obs2prior_dict': {'lambda_item': obs2prior, 'theta_user': False, 'alpha_item': obs2prior},
    #             'coef_dim_dict': {'lambda_item': 1, 'theta_user': 10, 'alpha_item': 10},
    #             'utility_formula': 'lambda_item + theta_user * alpha_item',
    #             'prior_variance': prior_std**2,
    #         })

    # =======
    hparam_list.append(
        {
            'obs2prior_dict': {
                'lambda_item': True,
                'theta_user': False,
                'alpha_item': True,
                'eta_user': False},
            'coef_dim_dict': {
                'lambda_item': 1,
                'theta_user': 5,
                'alpha_item': 5,
                'eta_user': num_item_obs},
            'utility_formula': 'lambda_item + theta_user * alpha_item + eta_user * item_obs',
            'prior_variance': 100,
        })

    # =======
    # for NUM_LATENTS in [3, 5, 10, 20]:
    #     hparam_list.append({
    #         'obs2prior_dict': {'lambda_item': True, 'theta_user': False, 'alpha_item': True, 'eta_user': False},
    #         'coef_dim_dict': {'lambda_item': 1, 'theta_user': NUM_LATENTS, 'alpha_item': NUM_LATENTS, 'eta_user': num_item_obs},
    #         'utility_formula': 'lambda_item + theta_user * alpha_item + eta_user * item_obs',
    #         'prior_variance': 10,
    #     })

    # =======
    for NUM_LATENTS in [3, 5, 10, 20]:
        hparam_list.append({
            'obs2prior_dict': {'lambda_item': True, 'theta_user': False, 'alpha_item': True},
            'coef_dim_dict': {'lambda_item': 1, 'theta_user': NUM_LATENTS, 'alpha_item': NUM_LATENTS},
            'utility_formula': 'lambda_item + theta_user * alpha_item',
            'prior_variance': 10,
        })

    for hparam in hparam_list:
        bemb = compute(hparam)
        # torch.save(bemb.state_dict(), './bemb.state_dict.pt')


if __name__ == '__main__':
    print('=' * 80)
    print('=' * 80)
    print('=' * 80)
    print('=' * 80)
    dataset = load_data(None, None)
    bemb = main(dataset)

import numpy as np
import torch
from bemb.model import LitBEMBFlex
from bemb.utils.run_helper import run

from load_data import load_data
from simulate_data import simulate_dataset


def main(dataset):
    num_users = len(torch.unique(dataset.user_index))
    num_items = len(torch.unique(dataset.item_index))
    num_item_obs = dataset.item_obs.shape[-1]

    # ----
    # for debugging only, run on a smaller sample.
    # dataset = dataset[np.random.permutation(len(dataset))[:100000]]
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
    obs2prior_dict = {'lambda_user': False, 'lambda_item': False, 'theta_user': False, 'alpha_item': False, 'eta_user': False}
    LATENT_DIM = 20
    coef_dim_dict = {'lambda_user': 1, 'lambda_item': 1, 'theta_user': LATENT_DIM, 'alpha_item': LATENT_DIM, 'eta_user': num_item_obs}

    bemb = LitBEMBFlex(
        # trainings args.
        learning_rate=0.03,
        pred_item=False,
        num_seeds=64,
        # model args, will be passed to BEMB constructor.
        # utility_formula='lambda_item + theta_user * alpha_item + eta_user * item_obs',
        # utility_formula='theta_user * alpha_item',
        utility_formula='theta_user * alpha_item',
        num_users=num_users,
        num_items=num_items,
        obs2prior_dict=obs2prior_dict,
        coef_dim_dict=coef_dim_dict,
        trace_log_q=True,
        num_item_obs=num_item_obs,
        prior_variance=10
    )

    bemb = bemb.to('cuda')
    bemb = run(bemb, dataset_list, batch_size=len(dataset) // 20, num_epochs=10)
    return bemb


if __name__ == '__main__':
    dataset = load_data()
    # dataset = simulate_dataset()
    bemb = main(dataset)

    state_dict = bemb.state_dict()

    torch.save(bemb.state_dict(), './bemb.state_dict.pt')

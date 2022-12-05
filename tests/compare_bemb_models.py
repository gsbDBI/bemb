"""
Compares the parameter estimation from BEMB in torch and C++.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from numpy import minimum
from scipy.stats import ttest_ind
from torch.nn.functional import log_softmax
from torch_scatter import scatter_max
from scipy.stats import norm
from torch_scatter.composite import scatter_softmax
import seaborn as sns

# cpp_path = '/home/tianyudu/Data/MoreSupermarket/t79338-n2123-m1109-k3-users3-lik3-shuffle0-eta0.005-zF0.1-nS10-batch5000-run_K3_500'
# cpp_path = '/home/tianyudu/Data/MoreSupermarket/t79338-n2123-m1109-k3-users3-lik3-shuffle0-eta0.005-zF0.1-nS10-batch5000-run_K3'
# cpp_path = '/home/tianyudu/Data/MoreSupermarket/t79338-n2123-m1109-k3-users3-lik3-shuffle0-eta0.005-zF0.1-nS10-batch5000-run_K3_500_rmsprop'
cpp_path = '/home/tianyudu/Data/MoreSupermarket/t79338-n2123-m1109-k20-users3-lik3-shuffle0-eta0.005-zF0.1-nS10-batch100000-run_K20_1000_rmsprop_bs100000'
torch_path = './output_K3/'


def plot_percentage_diff(A, B):
    A = A.reshape(-1,).detach().numpy()
    B = B.reshape(-1,).detach().numpy()
    plt.xlim((-1, 1))
    plt.hist((A - B) / B, bins=40)


@torch.no_grad()
def scale(x):
    # normalize utilty for items.
    # (num_users, num_items):
    mn = x.min(dim=1)[0].view(-1, 1)
    rg = (x.max(dim=1)[0] - x.min(dim=1)[0]).view(-1, 1)
    return (x - mn) / rg


def load_cpp_tsv(file):
    df = pd.read_csv(
        os.path.join(
            cpp_path,
            file),
        sep='\t',
        index_col=0,
        header=None)
    return torch.Tensor(df.values[:, 1:])


@torch.no_grad()
def compute_inner_product(user_latent, item_latent, item_to_category):
    x = user_latent @ item_latent.T  # (num_users, num_items)
    # compute probability of choosing each item from the respective category.
    # (num_users, num_categories)
    x = scatter_softmax(x, item_to_category, dim=-1)
    return x


def describe_diff(A, B):
    diff = A - B
    print('min(|A - B|)=', diff.abs().min().item())
    print('mean(|A - B|)=', diff.abs().mean().item())
    print('median(|A - B|)=', diff.abs().mean().item())
    print('max(|A - B|)=', diff.abs().max().item())


if __name__ == '__main__':
    # set empty string to use the last result. set to 100_ to load the params
    # after 100 iterations.
    it = ''

    # theta_u for users
    # alpha_i for items

    cpp_theta_mean = load_cpp_tsv(f'param_theta_{it}mean.tsv')
    cpp_alpha_mean = load_cpp_tsv(f'param_alpha_{it}mean.tsv')

    num_users = len(cpp_theta_mean)
    num_items = len(cpp_alpha_mean)

    print(f'{num_users=:} and {num_items=:}.')

    # ==============================================================================================
    #
    # ==============================================================================================

    torch_theta_mean = torch.load(os.path.join(torch_path, 'theta.pt')).cpu()
    torch_alpha_mean = torch.load(os.path.join(torch_path, 'alpha.pt')).cpu()

    # make sure dimensions are the same.
    # assert torch_theta_mean.shape[1] == cpp_theta_mean.shape[1]

    # group information.
    item_to_category = torch.load(
        os.path.join(
            torch_path,
            'category_idx.pt')).cpu()

    # inner product.
    cpp_ip = compute_inner_product(
        cpp_theta_mean,
        cpp_alpha_mean,
        item_to_category)
    torch_ip = compute_inner_product(
        torch_theta_mean,
        torch_alpha_mean,
        item_to_category)

    # ==============================================================================================
    # get the top prediction from each category.
    # ==============================================================================================
    output, counts = torch.unique(item_to_category, return_counts=True)
    num_categories = len(output)

    # random_ip = (1 / counts).view(1, num_categories).expand(num_users, -1)
    random_ip = scatter_softmax(torch.ones_like(cpp_ip), item_to_category)

    describe_diff(cpp_ip, torch_ip)
    describe_diff(cpp_ip, random_ip)

    fig, ax = plt.subplots()
    # ax.hist((cpp_ip - torch_ip).abs().view(-1), bins=100, label='C++ vs PyTorch', alpha=0.3)
    # ax.hist((cpp_ip - random_ip).abs().view(-1), bins=100, label='C++ vs Random', alpha=0.3)
    # ax.hist((torch_ip - random_ip).abs().view(-1), bins=100, label='PyTorch vs Random', alpha=0.3)
    diff = ((cpp_ip - torch_ip) / random_ip).view(-1)
    sns.distplot(diff, ax=ax, label='C++ vs PyTorch', bins=1000)
    diff = ((cpp_ip - random_ip) / random_ip).view(-1)
    sns.distplot(diff, ax=ax, label='C++ vs Random', bins=1000)
    diff = ((torch_ip - random_ip) / random_ip).view(-1)
    sns.distplot(diff, ax=ax, label='Pytorch vs Random', bins=1000)
    ax.set_xlim(-diff.quantile(0.99).item(), diff.quantile(0.99).item())
    plt.legend()
    plt.show()

    _, cpp_argmax = scatter_max(cpp_ip, item_to_category, dim=-1)
    _, torch_argmax = scatter_max(torch_ip, item_to_category, dim=-1)

    acc = (cpp_argmax == torch_argmax).float().mean()
    print(acc)

    # t-test
    torch_inner_product = torch_ip.numpy()
    cpp_inner_product = cpp_ip.numpy()

    pvals = list()
    for i in range(torch_inner_product.shape[0]):
        result = ttest_ind(torch_inner_product[i, :], cpp_inner_product[i, :])
        pvals.append(result.pvalue)

    plt.hist(pvals, bins=40)

    pvals = list()
    for i in range(torch_inner_product.shape[1]):
        result = ttest_ind(torch_inner_product[:, i], cpp_inner_product[:, i])
        pvals.append(result.pvalue)

    plt.hist(pvals, bins=40)

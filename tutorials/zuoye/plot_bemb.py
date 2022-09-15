import matplotlib.pyplot as plt
import argparse
import os

import numpy as np
import pandas as pd
import torch
from bemb.model import LitBEMBFlex
from bemb.utils.run_helper import run

from load_data import load_data
import seaborn as sns
from scipy import spatial
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


def plot_obs2prior(state_dict):
    pass


NUM_QUESTIONS = L.shape[0]
group = dict(zip(range(NUM_QUESTIONS),
                 [0] * (NUM_QUESTIONS // 2) + [1] * (NUM_QUESTIONS // 2)))
color = ['r'] * (NUM_QUESTIONS // 2) + ['b'] * (NUM_QUESTIONS // 2)


def plot_latent(state_dict):
    for coef_name in ['alpha_item']:
        L = state_dict[f'model.coef_dict.{coef_name}.variational_mean_flexible']
        norm = torch.sqrt(L.norm(dim=1).view(-1, 1)
                          * L.norm(dim=1).view(1, -1))
        data = L @ L.T / norm
        # data = L @ L.T
        g = sns.clustermap(data.numpy(), cmap='coolwarm', row_colors=color)
        g.savefig(f'./out/{coef_name}.png')

    L /= torch.sqrt(L.norm(dim=1).view(-1, 1))
    X_embedded = TSNE(n_components=2, init='random').fit_transform(L.numpy())
    attribute_path = '/home/tianyudu/Data/Zuoye/bayes/exam_response_ques_attrib.feather'
    response_path = '/home/tianyudu/Data/Zuoye/bayes/exam_response_with_attrib.feather'

    df_attr = pd.read_feather(attribute_path).sort_values(
        'question_id').reset_index(drop=True)
    df_resp = pd.read_feather(response_path)

    fig, ax = plt.subplots()
    # get question subject.
    df_subject = df_resp.groupby('question_id')['grade'].first()
    color = LabelEncoder().fit_transform(df_subject.values)
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1], ax=ax, c=color, alpha=1.0, s=5)
    fig.savefig('./out/alpha_item_tnse.png')

    # g.savefig('./out/theta_user.png')


if __name__ == '__main__':
    state_dict = torch.load('./bemb.state_dict.5d.pt')

    plot_obs2prior(bemb.model)

    plot_latent(bemb.model)

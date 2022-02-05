"""
See
https://cran.r-project.org/web/packages/mlogit/vignettes/e2nlogit.html
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from _deepchoice.data import ChoiceDataset, JointDataset, utils
from _deepchoice.model.nested_logit_model import NestedLogitModel
from _deepchoice.utils.std import parameter_std


def main(args):
    df = pd.read_csv(os.path.join(args.data_path, 'HC.csv'), index_col=0)
    df = df.reset_index(drop=True)
    session_ids = np.sort(df['idx.id1'].unique())

    # label
    # what was actually chosen.
    label = df[df['depvar'] == True].sort_values(by='idx.id1')['idx.id2'].reset_index(drop=True)
    # item_names = sorted(set(label.values))

    item_names = ['ec', 'ecc', 'er', 'erc', 'gc', 'gcc', 'hpc']
    num_items = df['idx.id2'].nunique()
    # cardinal encoder.
    encoder = dict(zip(item_names, range(num_items)))
    label = label.map(lambda x: encoder[x])
    label = torch.LongTensor(label)

    # unique session ids.
    session_ids = np.sort(df['idx.id1'].unique())

    # category feature: no category feature.
    # HOW TO DO THIS? DO THE INTERCEPT?
    category_dataset = ChoiceDataset(label=label.clone()).to('cuda')

    # item feature.
    item_feat_cols = ['ich', 'och', 'icca', 'occa', 'inc.room', 'inc.cooling', 'int.cooling']
    price_obs = utils.pivot3d(df, dim0='idx.id1', dim1='idx.id2', values=item_feat_cols)
    item_dataset = ChoiceDataset(label=label, price_obs=price_obs).to('cuda')

    # combine dataet
    dataset = JointDataset(category=category_dataset, item=item_dataset)

    data_loader = utils.create_data_loader(dataset, args)

    # build the model.
    category_to_item = {0: ['ec', 'ecc', 'gc', 'gcc', 'hpc'],
                        1: ['er', 'erc']}
    for k, v in category_to_item.items():
        v = [encoder[item] for item in v]
        category_to_item[k] = sorted(v)

    model = NestedLogitModel(category_to_item=category_to_item,
                             category_coef_variation_dict={},
                             category_num_param_dict={},
                            #  category_num_param_dict={'intercept': 1},
                             item_coef_variation_dict={'price_obs': 'constant'},
                             item_num_param_dict={'price_obs': 7}
                             )

    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9999)

    for e in range(10000):
        for batch in data_loader:
            loss = model.negative_log_likelihood(batch, batch['item'].label)
            if e % 100 == 0:
                print(f'{e=:}: {loss=:}, {model.lambda_constant.item()=:}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # model.clamp_lambdas()
        scheduler.step()
    print(f'{model.item_coef_dict.price_obs.coef=:}')

    # compute for Hessian.
    def nll_loss(model):
        # TODO: change batch to dataset.
        return model.negative_log_likelihood(batch, batch['item'].label)

    std = parameter_std(model, nll_loss)
    print(std)
    breakpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    args = parser.parse_args()

    args = argparse.Namespace(data_path='~/Development/deepchoice/data',
                              batch_size=-1,
                              shuffle=False,
                              device='cuda')
    main(args)

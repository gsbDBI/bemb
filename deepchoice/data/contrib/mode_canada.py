"""
Example dataloader of the mode canada dataset.
#TODO(Tianyu): add reference to this dataset.
"""
import os

import pandas as pd
import torch

from deepchoice.data import utils
from deepchoice.data import ChoiceDataset


def main(args) -> ChoiceDataset:
    mode_canada = pd.read_csv(os.path.join(args.data_path, 'ModeCanada.csv'), index_col=0)
    mode_canada = mode_canada.query('noalt == 4').reset_index(drop=True)

    num_sessions = mode_canada['case'].nunique()
    print(f'{num_sessions=:}')

    price_cost = utils.pivot3d(mode_canada, 'case', 'alt', ['cost', 'freq', 'ovt'])
    session_income = torch.Tensor(mode_canada.groupby('case')['income'].first().values).view(-1, 1)
    price_ivt = utils.pivot3d(mode_canada, 'case', 'alt', 'ivt')
    
    label = mode_canada.pivot('case', 'alt', 'choice')[['air', 'bus', 'car', 'train']].values
    _, label = torch.nonzero(torch.Tensor(label), as_tuple=True)
    label = label.long()
    
    user_onehot = torch.ones(num_sessions, 1).long()

    dataset = ChoiceDataset(label=label, user_onehot=user_onehot,
                            price_cost=price_cost,
                            price_ivt=price_ivt,
                            session_income=session_income).to(args.device)
    return dataset

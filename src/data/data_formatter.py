"""
This script contains utility functions to convert between different formats.
Please refer to the documentation for formats supported by this package.

Author: Tianyu Du
Date: July 11, 2021
"""
import os
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import torch


def tensors_to_stata() -> pd.DataFrame:
    
    raise NotImplementedError


def stata_to_tensors(df: pd.DataFrame,
                     session_id: str='id',
                     item_id: str='mode',
                     choice: str='choice',
                     user_id: Optional[str]=None,
                     user_vars: Optional[List[str]]=None,
                     case_vars: Optional[List[str]]=None,
                     item_vars: Optional[List[str]]=None
                     ) -> Dict[str, torch.Tensor]:
    """
    Converts Stata format of data to a list of tensors consisting of
    
    user_vars: variables depend only on i.
    case_vars: variables depend on i and t.
    item_vars: depends on i, j and t (user-specific price)
    """
    # example on the travel dataset:
    session_id = 'id'
    item_id = 'mode'
    category_id = None
    choice = 'choice'
    user_id = None
    df['user_id'] = 0  # treat all users as a meta-user, the user-agnostic formulation.

    # TODO(Tianyu): Do we need to distinguish these? 
    # u: user, i: item, t: time/session.
    var_cols_dict = {
        'u': None,
        'i': None,
        'ui': None,
        't': ['income', 'partysize'],
        'ut': None,
        'it': ['termtime', 'invehiclecost', 'traveltime', 'travelcost'],
        'uit': None
    }

    all_items = sorted(df[item_id].unique())  # NOTE: the order here is important.
    num_items = len(all_items)
    num_sessions = df[session_id].nunique()

    x_dict = dict()
    # reshape everything to (batch_size, num_items, num_params), num_params varies by tensor.
    for var_type in var_cols_dict.keys():
        x_sess_lst = list()
        a_lst = list()
        
        if var_cols_dict[var_type] is None:
            x_dict[var_type] = None
            continue
        
        num_params = len(var_cols_dict[var_type])
        for sess, sess_indices in df.groupby(session_id).indices.items():
            df_sess = df.loc[sess_indices]
            cols = var_cols_dict[var_type]  # columns for variables with this variation.
            # pivot, use for loop for readability.
            item_feat = list()
            item_avilability = torch.zeros(num_items)
            for i, item in enumerate(all_items):
                if item not in df_sess[item_id].values:
                    # the item is unavilable.
                    item_avilability[i] = 0
                    item_feat.append(torch.zeros(num_params))  # this is just a placeholder, does not matter.
                else:
                    # the item is available.
                    item_avilability[i] = 1
                    # retrieve relevant features in `cols` of this item in current session.
                    v = df_sess[df_sess[item_id] == item][cols].values.reshape(-1,)
                    item_feat.append(torch.Tensor(v))
            item_feat = torch.stack(item_feat)
            
            x_sess_lst.append(item_feat)
            a_lst.append(item_avilability)
        
        x_reshaped = torch.stack(x_sess_lst, dim=0)
        assert x_reshaped.shape == (num_sessions, num_items, num_params)
        
        a = torch.stack(a_lst, dim=0)
        assert a.shape == (num_sessions, num_items)

        x_dict[var_type] = x_reshaped
        x_dict['aviliability'] = a 

    # report tensors loaded.
    print('Features X, u=user, i=item, t=time/session.')
    for var_type, x in x_dict.items():
        try:
            print(f'X[{var_type}] with shape {x.shape}')
        except AttributeError:
            print(f'X[{var_type}] is None')
    return x_dict


def stata_to_X_Y_all(df: pd.DataFrame) -> Tuple[torch.Tensor]:
    raise NotImplementedError


if __name__ == '__main__':
    # Example usages.
    project_path = '/home/tianyudu/Development/deepchoice'
    travel = pd.read_csv(os.path.join(project_path, 'data/stata/travel.csv'))

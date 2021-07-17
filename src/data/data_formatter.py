"""
This script contains utility functions to convert between different formats.
Please refer to the documentation for formats supported by this package.

Author: Tianyu Du
Date: July 11, 2021
"""
import os
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch


def tensors_to_stata() -> pd.DataFrame:
    
    raise NotImplementedError


def stata_to_tensors(df: pd.DataFrame,
                     var_cols_dict: Dict[str, Union[List[str], None]],
                     session_id: str='id',
                     item_id: str='mode',
                     choice: str='choice',
                     category_id: Optional[str]=None,
                     user_id: Optional[str]=None
                     ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Converts Stata format of data to a dictionary of feature tensors and aviliability.

    Args:
        df (pd.DataFrame): the main dataframe in Stata's long format.
        var_cols_dict (Dict[str, Union[List[str], None]]): a dictionary with keys from ['u', 'i', 'ui', 't', 'ut', 'it', 'uit']
            and has list of column names in df as values.
            For example, var_cols_dict['u'] is the list of column names in df that are user-specific variables.
        session_id (str, optional): the column in df identifying session/trip ID.
            Defaults to 'id'.
        item_id (str, optional): the column in df identifying item ID.
            Defaults to 'mode'.
        choice (str, optional): the column in df identifying the chosen item in each session.
            Defaults to 'choice'.
        category_id (Optional[str], optional): the column in df identifying which category the item in
            each row belongs to.
            Defaults to None, set to None if all items belong to the same category.
        user_id (Optional[str], optional): the column in df identifying which user was involved in
            the session associated with each row.
            Defaults to None, set to None if all sessions are done by the same user.

    Returns:
        x_dict (Dict[str, torch.Tensor]): dictionary with keys from ['u', 'i', 'ui', 't', 'ut', 'it', 'uit'] and 'aviliability'.
            For variation keys like 'u' and 'i', out['u'] has shape (num_trips, num_items, num_params)
            as values.
        aviliability (torch.Tensor): a tensor with 0/1 values and has shape (num_trips, num_items)
            indicating the aviliability of each item during each shopping session.
        y (torch.Tensor): a tensor with shape (num_trips) indicating which one among all possible values
            in df[item_id] is chosen in that session.
    """
    all_items = sorted(df[item_id].unique())  # NOTE: the order here is important.
    num_items = len(all_items)
    # we don't assume the data type of raw item_id, we know convert to 
    item2int = dict(zip(all_items, range(num_items)))
    num_sessions = df[session_id].nunique()

    # load covariate tensors, each has shape (batch_size, num_items, num_params), num_params varies by tensor. 
    x_dict = dict()
    for var_type in var_cols_dict.keys():
        x_sess_lst = list()
        
        if var_cols_dict[var_type] is None:
            # there is no column corresponding to this type of variations. 
            x_dict[var_type] = None
            continue
        
        num_params = len(var_cols_dict[var_type])
        for sess, sess_indices in df.groupby(session_id).indices.items():
            df_sess = df.loc[sess_indices]  # rows in df corresponding to this session.
            cols = var_cols_dict[var_type]  # columns for variables with this variation.
            item_feat = list()  # item features in the current session, they are identical if the current x does not depend on item. 
            for i, item in enumerate(all_items):
                if item not in df_sess[item_id].values:
                    # the item is unavilable, load placeholder for features, we won't use them anyways.
                    item_feat.append(torch.zeros(num_params))
                else:
                    # retrieve relevant features in `cols` of this item in current session.
                    v = df_sess[df_sess[item_id] == item][cols].values.reshape(-1,)
                    item_feat.append(torch.Tensor(v))
            # item features for current session.
            item_feat = torch.stack(item_feat)
        
            assert item_feat.shape == (num_items, num_params)
            x_sess_lst.append(item_feat)
        
        x_reshaped = torch.stack(x_sess_lst, dim=0)
        assert x_reshaped.shape == (num_sessions, num_items, num_params)
        
        x_dict[var_type] = x_reshaped

    a_lst = list()  # aviliability info in each session.
    y_lst = list()  # choice info in each session.
            
    for sess, sess_indices in df.groupby(session_id).indices.items():
        df_sess = df.loc[sess_indices]
        
        item_avilability = torch.zeros(num_items)
        for i, item in enumerate(all_items):
            item_avilability[i] = 1 if item in df_sess[item_id].values else 0
        a_lst.append(item_avilability)
        
        c = df_sess[df_sess[choice] == 1][item_id]
        assert len(c) == 1  # only 1 item should be chosen.
        y_lst.append(item2int[c.iloc[0]])  # append the code of chosen item.

    a = torch.stack(a_lst, dim=0)
    assert a.shape == (num_sessions, num_items)

    y = torch.Tensor(y_lst)
    assert y.shape == (num_sessions,)

    # report tensors loaded.
    print('Features X loaded, u=user, i=item, t=time/session.')
    for var_type, x in x_dict.items():
        try:
            print(f'X[{var_type}] with shape {x.shape}')
        except AttributeError:
            print(f'X[{var_type}] is None')
            
    return x_dict, a, y


def stata_to_X_Y_all(df: pd.DataFrame) -> Tuple[torch.Tensor]:
    raise NotImplementedError


if __name__ == '__main__':
    # Example usages.
    project_path = '/home/tianyudu/Development/deepchoice'
    travel = pd.read_csv(os.path.join(project_path, 'data/stata/travel.csv'))
    # example on the travel dataset:
    session_id = 'id'
    item_id = 'mode'
    choice = 'choice'
    category_id = None
    user_id = None

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
    
    X, A, Y = stata_to_tensors(df=travel,
                               var_cols_dict=var_cols_dict,
                               session_id='id',
                               item_id='mode',
                               choice='choice',
                               category_id=None,
                               user_id=None)

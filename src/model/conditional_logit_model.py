"""
Conditional Logit Model, the generalized version of the `cmclogit' command in Stata.

Author: Tianyu Du
Date: Jul. 11, 2021
"""
from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn as nn


class Coefficient(nn.Module):
    def __init__(self,
                 variation: str,
                 num_items: int,
                 num_users: Optional[int]=None,
                 num_params: Optional[int]=None) -> None:
        """A generic coefficient object storing trainable parameters.

        Args:
            variation (str): [description]
            num_items (int): [description]
            num_users (Optional[int], optional): [description]. Defaults to None.
            num_params (Optional[int], optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]
        """
        super(Coefficient, self).__init__()
        assert variation in ['zero', 'constant', 'item', 'user', 'user-item']
        self.variation = variation
        self.num_items = num_items
        self.num_users = num_users
        self.num_params = num_params

        if self.variation == 'constant':
            # constant for all users and items.
            self.coef = nn.Parameter(torch.randn(num_params), requires_grad=True)
        elif self.variation == 'item':
            # coef_j depends on item j but not on user i.
            self.coef = nn.Parameter(torch.randn(num_items, num_params), requires_grad=True)
        elif self.variation == 'user':
            self.coef = nn.Parameter(torch.randn(num_users, num_params), requires_grad=True)
        elif self.variation == 'user-item':
            self.coef = nn.Parameter(torch.randn(num_users, num_items, num_params), requires_grad=True)
        else:
            raise ValueError(f'Unsupported type of variation: {self.variation}.')

    def __repr__(self):
        return f'Coefficient({self.variation}, {self.num_params} parameters for each user-item, total {self.coef.numel()} trainable parameters)'

    def forward(self, x: torch.Tensor, user_onehot: Optional[torch.Tensor]=None):
        num_trips, num_items, num_feats = x.shape
        assert self.num_params == num_feats
        if self.variation == 'constant':
            coef = self.coef.expand(num_trips, num_items, -1)
            return (x * coef).sum(dim=-1)
        elif self.variation == 'item':
            coef = self.coef.expand(num_trips, -1, -1)
            return (x * coef).sum(dim=-1)
        elif self.variation == 'user':
            coef_user = user_onehot @ self.coef  # (num_trips, num_params)
            coef_user = coef_user.expand(-1, num_items, -1)  # (num_trips, num_items, num_params)
            return (coef_user * x).sum(dim=-1)
        elif self.variation == 'user-item':
            # look up the coef corresponding to current user.
            user_idx = torch.nonzero(user_onehot, as_tuple=True)[1]
            coef_user_item = self.coef[user_idx, :, :]  # (num_trips, num_items, num_params)
            assert coef_user_item == x.shape
            # collapse along the parameter axis.
            # output[trip, j] = <coef[user_i, item_j], feat[user_i, item_j]>  # inner product.
            return (coef_user_item * x).sum(dim=-1)  # (num_trips, num_items)
        else:
            raise ValueError(f'Unsupported type of variation: {self.variation}.')


class ConditionalLogitModel(nn.Module):
    """The more generalized version of conditional logit model, the model allows for 1+7 types of
    variables, including:
    u: user-specific features,
    i: item-specific features,
    ui: user-item specific features,
    {t, ut, it, uit}: {time, user-time, item-time, user-item-time} specific features.
    intercept term.
    
    NOTE: the order of u, i, and t matters.

    The model allows for
    constant: constant over all users and items,
    user: user-specific parameters but constant across all items,
    item: item-specific parameters but constant across all users,
    user-item: parameters that are specific to both user and item,
    variation for each type of variables.
    """
    def __init__(self,
                 num_items: int,
                 num_users: int,
                 var_variation_dict: Dict[str, str],
                 var_num_params_dict: Dict[str, int]
                 ):
        """[summary]

        Args:
            num_items (int): number of items in the dataset.
            num_users (int): number of users in the dataset.
            var_variation_dict (Dict[str, str]): variable type to variation level dictionary.
                Put None or 'zero' if there is no this kind of variable in the model.
            var_num_params_dict (Dict[str, int]): variable type to number of parameters dictionary,
                records number of features in each kind of variable.
                Put None if there is no this kind of variable in the model.
        """
        super(ConditionalLogitModel, self).__init__()
        ALLOWED_VARIABLES = ['intercept', 'u', 'i', 'ui', 't', 'ut', 'it', 'uit'] 
        ALLOWED_VARIATION_LEVELS = [None, 'constant', 'user', 'item', 'user-item']
        
        self.var_num_params_dict = deepcopy(var_num_params_dict)
        # check number of parameters specified are all positive.
        for var_type, num_params in self.var_num_params_dict.items():
            assert var_type in ALLOWED_VARIABLES, f'variable type {var_type} is now allowed.'
            if num_params is not None:
                assert num_params > 0, 'num_params needs to be positive if specified.'
        # intercept only comes with 1 parameter, add it if not provided.
        if 'intercept' not in self.var_num_params_dict.keys():
            self.var_num_params_dict['intercept'] = 1

        # check variation levels specified for each type of variables.
        self.var_variation_dict = deepcopy(var_variation_dict)
        for var_type, variation in self.var_variation_dict.items():
            assert var_type in ALLOWED_VARIABLES, f'variable type {var_type} is now allowed.'
            assert variation in ALLOWED_VARIATION_LEVELS, f'variation type {variation} is not allowed.'

        self.num_items = num_items
        self.num_users = num_users
        
        # construct trainable parameters.
        coef_dict = dict()
        for var_type, variation in self.var_variation_dict.items():
            if variation is None:
                coef_dict[var_type] = None
            else:
                num_params = self.var_num_params_dict[var_type]
                coef_dict[var_type] = Coefficient(variation, self.num_items, self.num_users, num_params)

        # NOTE: a ModuleDict is required for self.parameters() to retrieve the correct weight.
        self.coef_dict = nn.ModuleDict(coef_dict)

    def __repr__(self) -> str:
        out_str_lst = ['Conditional logistic discrete choice model, expects input features:\n']
        for var_type, num_params in self.var_num_params_dict.items():
            if num_params is not None:
                out_str_lst.append(f'X[{var_type}] with {num_params} parameters, with {self.var_variation_dict[var_type]} level variation.')
        return super().__repr__() + '\n' + '\n'.join(out_str_lst)

    @property
    def total_params(self) -> int:
        return sum(w.numel() for w in self.parameters())

    def summary(self):
        for var_type, coefficient in self.coef_dict.items():
            if coefficient is not None:
                print(var_type)
                print(coefficient.coef)

    def forward(self,
                x_dict: Dict[str, torch.Tensor],
                availability: torch.Tensor,
                user_onehot: torch.Tensor=None,
                ) -> torch.Tensor:
        assert not all(x is None for x in x_dict.values()), 'No input tensor found.'

        # get input shapes.
        for x in x_dict.values():
            if x is not None:
                device = x.device
                batch_size = x.shape[0]
                break
        
        # compute the utility from each item in each choice session.
        total_utility = torch.zeros(batch_size, self.num_items).to(device)
        # intercept term has no input tensor, which has only 1 feature.
        x_dict['intercept'] = torch.ones(batch_size, self.num_items, 1).to(device)

        # for each type of variables, apply the corresponding coefficient to input x.
        for var_type, coef in self.coef_dict.items():
            if x_dict[var_type] is not None:
                total_utility += coef(x_dict[var_type], user_onehot)
    
        assert total_utility.shape == (batch_size, self.num_items)
        # mask out unavilable items.
        total_utility[~availability] = -1.0e20
        return total_utility

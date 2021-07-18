"""
Conditional Logit Model, the generalized version of the `cmclogit' command in Stata.

Author: Tianyu Du
Date: Jul. 11, 2021
"""
import torch
import torch.nn as nn
import math
from typing import Union, Optional, Dict
from torch.nn import functional as F


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
        super().__init__()
        self.variation = variation
        self.num_items = num_items
        self.num_users = num_users
        self.num_params = num_params

        if self.variation == 'zero':
            # this term virtually does not exist in the equation of utility.
            self.coef = None
        elif self.variation == 'constant':
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
        return f'{self.variation} specific linear coefficients.'

    def forward(self, x: torch.Tensor, user_onehot: Optional[torch.Tensor]=None):
        num_trips, num_items, num_feats = x.shape
        assert self.num_params == num_feats
        if self.variation == 'zero':
            return torch.zeros((num_trips, num_items), requires_grad=True)
        elif self.variation == 'constant':
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


# class ConditionalLogitModel(nn.Module):
#     def __init__(self,
#                  user_feature_dim: int,
#                  item_feature_dim: int,
#                  price_feature_dim: int,
#                  user_price_feature_dim: int,
#                  num_items: int,
#                  num_users: int,
#                  intercept_variation: str='constant',
#                  user_variation: str='item',
#                  item_variation: str='constant',
#                  price_variation: str='constant',
#                  user_price_variation: str='constant'
#                  ):
#         super().__init__()
#         self.user_feature_dim = user_feature_dim
#         self.item_feature_dim = item_feature_dim
#         self.price_feature_dim = price_feature_dim
#         self.user_price_feature_dim = user_price_feature_dim

#         self.num_items = num_items
#         self.num_users = num_users
        
#         self.intercept_variation = intercept_variation
#         self.user_variation = user_variation
#         self.item_variation = item_variation
#         self.price_variation = price_variation
#         self.user_price_variation = user_price_variation

#         # construct trainables.
#         self.intercept = Coefficient(self.intercept_variation,
#                                      self.num_items,
#                                      self.num_users,
#                                      num_params=1)
        
#         self.user_coef = Coefficient(self.user_variation,
#                                      self.num_items,
#                                      self.num_users,
#                                      self.user_feature_dim)
        
#         self.item_coef = Coefficient(self.item_variation,
#                                      self.num_items,
#                                      self.num_users,
#                                      self.item_feature_dim)
        
#         self.price_coef = Coefficient(self.price_variation,
#                                       self.num_items,
#                                       self.num_users,
#                                       self.price_feature_dim)
        
#         self.user_price_coef = Coefficient(self.user_price_variation,
#                                            self.num_items,
#                                            self.num_users,
#                                            self.user_price_feature_dim)

#     def __repr__(self):
#         def symbol(greek, variation):
#             if variation == 'zero':
#                 return '0'
#             elif variation == 'constant':
#                 return greek
#             elif variation == 'item':
#                 return greek + '[j]'
#             elif variation == 'user':
#                 return greek + '[i]'
#             else:
#                 return greek + '[i, j]'
#         model = f"{type(self)} object with utility expression" \
#                 + f"\n U[i,j,t] = {symbol('alpha', self.intercept_variation)}" \
#                 + f" + {symbol('beta', self.user_variation)} x_user[i]" \
#                 + f" + {symbol('gamma', self.item_variation)} x_item[j]" \
#                 + f" + {symbol('eta', self.price_variation)} x_price[j,t]" \
#                 + f" + {symbol('delta', self.user_price_variation)} x_user_price[i,j,t]" \
#                 + f" + epsilon[i,j,t]" \
#                 + f"\n Number of parameters = {self.num_params()}."
#         return model

#     def num_params(self) -> int:
#         total_params = 0
#         for weight in self.parameters():
#             total_params += weight.numel()
#         return total_params

#     def forward(self,
#                 x_user: torch.Tensor=None,
#                 user_onehot: torch.Tensor=None,
#                 x_item: torch.Tensor=None,
#                 x_price: torch.Tensor=None,
#                 x_user_price: torch.Tensor=None,
#                 availability: torch.Tensor=None) -> torch.Tensor:
#         """
#         NOTE: num_item = class_dim
        
#         expected shapes, where num_obs denotes number of trips.
#         x_user: num_obs x user_feature_dim
#         user_onehot: num_obs x num_users
#         x_item: num_item x item_feature_dim
#         x_price: num_obs x num_item x price_feature_dim
#         x_user_specific_price: num_obs x num_items x user_price_feature_dim
#         availability: num_obs x num_item, binary.
#         """
#         device = x_user.device()
#         batch_size = x_user.shape[0]
#         # expand tensors to (batch_size, num_items, num_params), num_params varies by args.
#         x_user = x_user.expand(-1, self.num_items, -1)
#         x_item = x_item.expand(batch_size, -1, -1)
        
#         ones = torch.ones(batch_size, self.num_users, 1)
#         utility_intercept = self.intercept(ones, user_onehot)
        
#         utility_user = self.user_coef(x_user, user_onehot)
        
#         utility_item = self.item_coef(x_item, user_onehot)
        
#         utility_price = self.item_coef(x_price, user_onehot)
        
#         utility_user_price = self.user_price_coef(x_user_price, user_onehot)
        
#         total_utility = utility_intercept + utility_user + utility_item + utility_price + utility_user_price
#         assert total_utility.shape == (batch_size, self.num_items)
        
#         # mask out unavilable items
#         total_utility[~availability] = -1.0e20
#         return total_utility


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
    zero: ignore this term,
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
        super().__init__()

        self.var_num_params_dict = var_num_params_dict
        for k, num_params in self.var_num_params_dict.items():
            assert k in ['u', 'i', 'ui', 't', 'ut', 'it', 'uit'], f'variable type {k} is now allowed.'
            assert num_params > 0 or num_params is None
        self.var_num_params_dict['intercept'] = 1

        self.var_variation_dict = var_variation_dict
        for k, v in self.var_variation_dict.items():
            assert k in ['intercept', 'u', 'i', 'ui', 't', 'ut', 'it', 'uit'], f'variable type {k} is now allowed.'
            assert v in [None, 'zero', 'constant', 'user', 'item', 'user-item'], f'variation type {v} is not allowed.'

        self.num_items = num_items
        self.num_users = num_users
        
        # construct trainable parameters.
        self.coef = dict()
        for var_type, variation in self.var_variation_dict.items():
            if variation is None:
                self.coef[k] = None
            else:
                num_params = self.var_num_params_dict[var_type]
                self.coef[k] = Coefficient(variation, self.num_items, self.num_users, num_params)

    def __repr__(self) -> str:
        out_str_lst = list()
        for var_type, num_params in self.var_num_params_dict.items():
            try:
                out_str_lst.append(f'X[{var_type}] with {num_params} parameters')
            except AttributeError:
                out_str_lst.append(f'X[{var_type}] is None')
        return '\n'.join(out_str_lst)

    def num_params(self) -> int:
        total_params = 0
        for weight in self.parameters():
            total_params += weight.numel()
        return total_params

    def forward(self,
                x_dict: Dict[str, torch.Tensor],
                availability: torch.Tensor,
                user_onehot: torch.Tensor=None,
                ) -> torch.Tensor:
        assert not all(x is None for x in x_dict.values()), 'No input tensor found.'

        # get input shapes.
        for x in x_dict.values():
            if x is not None:
                device = x.device()
                batch_size = x.shape[0]
                break
        
        # compute the utility from each item in each choice session.
        total_utility = torch.zeros(batch_size, self.num_items).to(device)
        # intercept term has no input tensor, which has only 1 feature.
        x_dict['intercept'] = torch.ones(batch_size, self.num_items, 1).to(device)

        # for each type of variables, apply the corresponding coefficient to input x.
        for var_type, coef in self.coef.items():
            if x_dict[var_type] is not None:
                total_utility += coef(x_dict[var_type], user_onehot)
    
        assert total_utility.shape == (batch_size, self.num_items)
        # mask out unavilable items.
        total_utility[~availability] = -1.0e20
        return total_utility

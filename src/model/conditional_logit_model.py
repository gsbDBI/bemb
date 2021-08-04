"""
Conditional Logit Model, the generalized version of the `cmclogit' command in Stata.

Author: Tianyu Du
Date: Jul. 11, 2021
"""
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Coefficient(nn.Module):
    def __init__(self,
                 variation: str,
                 num_items: int,
                 num_users: Optional[int]=None,
                 num_params: Optional[int]=None) -> None:
        """A generic coefficient object storing trainable parameters.

        Args:
            variation (str): the degree of variation of this coefficient. For example, the
                coefficient can vary by users or items.
            num_items (int): number of items.
            num_users (Optional[int], optional): number of users, this is only necessary if
                the coefficient varies by users.
                Defaults to None.
            num_params (Optional[int], optional): number of parameters. Defaults to None.
        """
        super(Coefficient, self).__init__()
        self.variation = variation
        self.num_items = num_items
        self.num_users = num_users
        self.num_params = num_params

        # construct the trainable.
        if self.variation == 'constant':
            # constant for all users and items.
            self.coef = nn.Parameter(torch.randn(num_params), requires_grad=True)
        elif self.variation == 'item':
            # coef depends on item j but not on user i.
            # force coefficeints for the first item class to be zero.
            self.coef = nn.Parameter(torch.zeros(num_items - 1, num_params), requires_grad=True)
        elif self.variation == 'item-full':
            # coef depends on item j but not on user i.
            # model coefficient for every item.
            self.coef = nn.Parameter(torch.zeros(num_items, num_params), requires_grad=True)
        elif self.variation == 'user':
            # coef depends on the user.
            # we always model coefficeints for all users.
            self.coef = nn.Parameter(torch.zeros(num_users, num_params), requires_grad=True)
        elif self.variation == 'user-item':
            # coefficients of the first item is forced to be zero, model coefficients for N - 1 items only.
            self.coef = nn.Parameter(torch.zeros(num_users, num_items - 1, num_params), requires_grad=True)
        elif self.variation == 'user-item-full':
            # construct coefficients for every items.
            self.coef = nn.Parameter(torch.zeros(num_users, num_items, num_params), requires_grad=True)
        else:
            raise ValueError(f'Unsupported type of variation: {self.variation}.')

    def __repr__(self):
        return f'Coefficient(variation={self.variation}, {self.num_params} parameters, total {self.coef.numel()} trainable parameters)'

    def forward(self,
                x: torch.Tensor,
                user_onehot: Optional[torch.Tensor]=None,
                manual_coef_value: Optional[torch.Tensor]=None
                ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): a tensor of shape (num_sessions, num_items, num_params).
            user_onehot (Optional[torch.Tensor], optional): a tensor of shape (num_sessions, num_users)
                in which one row is the one-hot vector of the user involved in that session.
                Defaults to None.
            manual_coef_value (Optional[torch.Tensor], optional): a tensor with the same number of
                entries as self.coef. If provided, the forward function uses provided values
                as coefficient and return the predicted utility, this feature is useful when
                the researcher wishes to manually specify vaues for coefficients and examine prediction
                with specified coefficient values. If not provided, forward function is executed
                using values from self.coef.
                Defaults to None.

        Returns:
            torch.Tensor: a tensor of shape (num_sessions, num_items) whose (t, i) entry represents
                the utility of purchasing item i in session t.
        """
        if manual_coef_value is not None:
            assert manual_coef_value.numel() == self.coef.numel()
            # plugin the provided coefficient values, coef is a tensor.
            coef = manual_coef_value.reshape(*self.coef.shape)
        else:
            # use the learned coefficient values, coef is a nn.Parameter.
            coef = self.coef
        
        num_trips, num_items, num_feats = x.shape
        assert self.num_params == num_feats

        # cast coefficient tensor to (num_trips, num_items, self.num_params).
        if self.variation == 'constant':
            coef = coef.view(1, 1, self.num_params).expand(num_trips, num_items, -1)

        elif self.variation == 'item':
            # coef has shape (num_items-1, num_params)
            # force coefficient for the first item to be zero.
            zeros = torch.zeros(1, self.num_params).to(coef.device)
            coef = torch.cat((zeros, coef), dim=0)  # (num_items, num_params)
            coef = coef.view(1, self.num_items, self.num_params).expand(num_trips, -1, -1)

        elif self.variation == 'item-full':
            # coef has shape (num_items, num_params)
            coef = coef.view(1, self.num_items, self.num_params).expand(num_trips, -1, -1)

        elif self.variation == 'user':
            # coef has shape (num_users, num_params)
            coef = user_onehot @ coef  # (num_trips, num_params) user-specific coefficients.
            coef = coef.view(num_trips, 1, self.num_params).expand(-1, num_items, -1)

        elif self.variation == 'user-item':
            # (num_trips,) long tensor of user ID.
            user_idx = torch.nonzero(user_onehot, as_tuple=True)[1]
            # originally, coef has shape (num_users, num_items-1, num_params)
            # transform to (num_trips, num_items - 1, num_params), user-specific.
            coef = coef[user_idx, :, :]
            # coefs for the first item for all users are enforced to 0.
            zeros = torch.zeros(num_trips, 1, self.num_params).to(coef.device)
            coef = torch.cat((zeros, coef), dim=1)  # (num_trips, num_items, num_params)

        elif self.variation == 'user-item-full':
            # originally, coef has shape (num_users, num_items, num_params)
            user_idx = torch.nonzero(user_onehot, as_tuple=True)[1]
            coef = coef[user_idx, :, :]  # (num_trips, num_items, num_params)

        else:
            raise ValueError(f'Unsupported type of variation: {self.variation}.')

        assert coef.shape == (num_trips, num_items, num_feats) == x.shape
        
        # compute the utility of each item in each trip.
        return (x * coef).sum(dim=-1)


class ConditionalLogitModel(nn.Module):
    """The more generalized version of conditional logit model, the model allows for 1+7 types of
    variables, including:
    - u: user-specific features,
    - i: item-specific features,
    - ui: user-item specific features,
    - similarly,{t, ut, it, uit}: {time, user-time, item-time, user-item-time} specific features.
    - intercept term.
    NOTE: the intercept term will NOT be added unless explicitly specified.

    NOTE: the order of u, i, and t matters.

    The model allows for the following levels for variable variations:
    NOTE: unless the `-full` flag is specified (which means we want to explicitly model coefficients
        for all items), for all variation levels related to item (item specific and user-item specific),
        the model force coefficients for the first item to be zero. This design follows stadnard
        econometric practice.

    - constant: constant over all users and items,
    
    - user: user-specific parameters but constant across all items,
    
    - item: item-specific parameters but constant across all users, parameters for the first item are
        forced to be zero.
    - item-full: item-specific parameters but constant across all users, explicitly model for all items.

    - user-item: parameters that are specific to both user and item, parameterts for the first item
        for all users are forced to be zero.
    - user-item-full: parameters that are specific to both user and item, explicitly model for all items.
    """
    def __init__(self,
                 num_items: int,
                 num_users: int,
                 var_variation_dict: Dict[str, str],
                 var_num_params_dict: Dict[str, int]
                 ) -> None:
        """
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
        ALLOWED_VARIABLE_TYPES = ['intercept', 'u', 'i', 'ui', 't', 'ut', 'it', 'uit']
        ALLOWED_VARIATION_LEVELS = ['constant', 'user', 'item', 'item-full', 'user-item', 'user-item-full']

        assert var_variation_dict.keys() == var_num_params_dict.keys()
        self.variable_types = list(deepcopy(var_variation_dict).keys())
        self.var_num_params_dict = deepcopy(var_num_params_dict)
        self.var_variation_dict = deepcopy(var_variation_dict)
        self.num_items = num_items
        self.num_users = num_users

        # check number of parameters specified are all positive.
        for var_type, num_params in self.var_num_params_dict.items():
            assert var_type in ALLOWED_VARIABLE_TYPES, f'variable type {var_type} is now allowed. Allowed types: {ALLOWED_VARIABLE_TYPES}'
            assert num_params > 0, f'num_params needs to be at least 1, got: {num_params}.'

        # check variation levels specified for each type of variables.
        for var_type, variation in self.var_variation_dict.items():
            assert var_type in ALLOWED_VARIABLE_TYPES, f'variable type {var_type} is now allowed. Allowed types: {ALLOWED_VARIABLE_TYPES}'
            assert variation in ALLOWED_VARIATION_LEVELS, f'variation type {variation} is not allowed. Allowed levels: {ALLOWED_VARIATION_LEVELS}'

        # infer the number of parameters for intercept if the researcher forgets.
        if 'intercept' in self.var_variation_dict.keys() and 'intercept' not in self.var_num_params_dict.keys():
            self.var_num_params_dict['intercept'] = 1

        # construct trainable parameters.
        coef_dict = dict()
        for var_type, variation in self.var_variation_dict.items():
            coef_dict[var_type] = Coefficient(variation=variation,
                                              num_items=self.num_items,
                                              num_users=self.num_users,
                                              num_params=self.var_num_params_dict[var_type])
        # A ModuleDict is required to properly register all trainiabel parameters.
        # self.parameter() will fail if a python dictionary is used instead.
        self.coef_dict = nn.ModuleDict(coef_dict)

    def __repr__(self) -> str:
        out_str_lst = ['Conditional logistic discrete choice model, expects input features:\n']
        for var_type, num_params in self.var_num_params_dict.items():
            out_str_lst.append(f'X[{var_type}] with {num_params} parameters, with {self.var_variation_dict[var_type]} level variation.')
        return super().__repr__() + '\n' + '\n'.join(out_str_lst)

    @property
    def total_params(self) -> int:
        return sum(w.numel() for w in self.parameters())

    def summary(self):
        for var_type, coefficient in self.coef_dict.items():
            if coefficient is not None:
                print('Variable Type: ', var_type)
                print(coefficient.coef)

    def forward(self,
                x_dict: Dict[str, torch.Tensor],
                availability: torch.BoolTensor=None,
                user_onehot: torch.Tensor=None,
                manual_coef_value_dict: Optional[Dict[str, torch.Tensor]]=None
                ) -> torch.Tensor:
        """
        Args:
            x_dict (Dict[str, torch.Tensor]): a dictionary where keys are in {'u', 'i'} etc, and
                values are tensors with shape (num_trips, num_items, num_params), where num_trips
                and num_items are the same for all values but num_params may vary.
            availability (torch.BoolTensor, optional): a boolean tensor with shape (num_trips, num_items)
                where A[t, i] indicates the aviliability of item i in trip t. Utility of unavilable
                items will be set to -inf. If not provided, the model assumes all items are aviliability.
                Defaults to None.
            user_onehot (torch.Tensor, optional): a tensor with shape (num_trips, num_users), the t-th
                row of this tensor is the one-hot vector indciating which user is involved in this trip.
                user_onehot is required only if there exists user-specifc or user-item-specific coefficients
                in the model.
                Defaults to None.
            manual_coef_value_dict (Optional[Dict[str, torch.Tensor]], optional): a dictionary with
                keys in {'u', 'i'} etc and tensors as values. If provided, the model will force
                coefficient to be the provided values and compute utility conditioned on the provided
                coefficient values. This feature is useful when the research wishes to plug in particular
                values of coefficients and exmaine the utility values. If not provided, the model will
                use the learned coefficient values in self.coef_dict. 
                Defaults to None.

        Returns:
            torch.Tensor: a tensor of shape (num_trips, num_items) whose (t, i) entry represents
                the utility from item i in trip t for the user involved in that trip. 
        """
        # get input shapes.
        for x in x_dict.values():
            if x is not None:
                device = x.device
                batch_size = x.shape[0]
                break

        if 'intercept' in self.var_variation_dict.keys():
            # intercept term has no input tensor, which has only 1 feature.
            x_dict['intercept'] = torch.ones(batch_size, self.num_items, 1).to(device)

        # compute the utility from each item in each choice session.
        total_utility = torch.zeros(batch_size, self.num_items).to(device)
        # for each type of variables, apply the corresponding coefficient to input x.
        for var_type, coef in self.coef_dict.items():
            if manual_coef_value_dict is not None:
                total_utility += coef(x_dict[var_type], user_onehot, manual_coef_value_dict[var_type])
            else:
                total_utility += coef(x_dict[var_type], user_onehot)
    
        assert total_utility.shape == (batch_size, self.num_items)

        if availability is not None:
            # mask out unavilable items.
            total_utility[~availability] = -1.0e20
        return total_utility

    @staticmethod
    def flatten_coef_dict(coef_dict: Dict[str, Union[torch.Tensor, torch.nn.Parameter]]) -> Tuple[torch.Tensor, dict]:
        """Flattens the coef_dict into a 1-dimension tensor, used for hessian computation."""
        type2idx = dict()
        param_list = list()
        start = 0

        for var_type in coef_dict.keys():
            num_params = coef_dict[var_type].coef.numel()
            # track which portion of all_param tensor belongs to this variable type.
            type2idx[var_type] = (start, start + num_params)
            start += num_params
            # use reshape instead of view to make a copy.
            param_list.append(coef_dict[var_type].coef.clone().reshape(-1,))

        all_param = torch.cat(param_list)  # (self.total_params(), )
        return all_param, type2idx

    @staticmethod
    def unwrap_coef_dict(param: torch.Tensor, type2idx: Dict[str, Tuple[int, int]]) -> Dict[str, torch.Tensor]:
        """Rebuild coef_dict from output of self.flatten_coef_dict method."""
        coef_dict = dict()
        for var_type in type2idx.keys():
            start, end = type2idx[var_type]
            # no need to reshape here, Coefficient handles it.
            coef_dict[var_type] = param[start:end]
        return coef_dict

    def compute_hessian(self, x_dict, availability, user_onehot, y) -> torch.Tensor:
        """Computes the hessian of negaitve log-likelihood (total cross-entropy loss) with respect
        to all parameters in this model.

        Args:
            x_dict ,availability, user_onehot: see definitions in self.forward.
            y (torch.LongTensor): a tensor with shape (num_trips,) of IDs of items actually purchased.

        Returns:
            torch.Tensor: a (self.total_params, self.total_params) tensor of the Hessian matrix.
        """
        all_coefs, type2idx = self.flatten_coef_dict(self.coef_dict)
        
        def compute_nll(P: torch.Tensor) -> float:
            coef_dict = self.unwrap_coef_dict(P, type2idx)
            y_pred = self.forward(x_dict=x_dict,
                                  availability=availability,
                                  user_onehot=user_onehot,
                                  manual_coef_value_dict=coef_dict)
            # the reduction needs to be 'sum' to obtain NLL.
            loss = F.cross_entropy(y_pred, y, reduction='sum')
            return loss

        H = torch.autograd.functional.hessian(compute_nll, all_coefs)
        assert H.shape == (self.total_params, self.total_params)
        return H

    # def compute_approx_hessian(self, x_dict, availability, user_onehot, y):
    #     self.zero_grad()
    #     y_pred = self.forward(x_dict, availability, user_onehot)
    #     loss = F.cross_entropy(y_pred, y, reduction='sum')
    #     loss.backward()
        
    #     grad_list = list()
    #     for var_type in self.coef_dict.keys():
    #         atomic_grad = self.coef_dict[var_type].coef.grad
    #         grad_list.append(atomic_grad.view(-1,))
    #     grad = torch.cat(grad_list).view(-1, 1)
    #     H = grad @ grad.T
    #     return H

    def compute_std(self, x_dict, availability, user_onehot, y) -> Dict[str, torch.Tensor]:
        """Computes

        Args:
            See definitions in self.compute_hessian.

        Returns:
            Dict[str, torch.Tensor]: a dictoinary whose keys are the same as self.coef_dict.keys()
            the values are standard errors of coefficients in each coefficient group. 
        """
        _, type2idx = self.flatten_coef_dict(self.coef_dict)
        H = self.compute_hessian(x_dict, availability, user_onehot, y)
        std_all = torch.sqrt(torch.diag(torch.inverse(H)))
        std_dict = dict()
        for var_type in type2idx.keys():
            # get std of variables belonging to each type.
            start, end = type2idx[var_type]
            std_dict[var_type] = std_all[start:end]
        return std_dict

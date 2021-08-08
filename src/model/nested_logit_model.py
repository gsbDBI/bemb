from typing import Dict, List

import torch
import torch.nn as nn
import warnings


class NestedLogitModel(nn.Module):
    # The nested logit model.
    def __init__(self,
                 category_feature_dim: int,
                 item_feature_dim: int,
                 category_to_item: Dict[object, List[int]]
                 ) -> None:
        """"""
        super(NestedLogitModel, self).__init__()
        
        self.category_feature_dim = category_feature_dim
        self.item_feature_dim = item_feature_dim
        
        self.category_to_item = category_to_item
        self.num_categories = len(category_to_item)
        self.categories = list(category_to_item.keys())
        self.num_items = sum(len(items) for items in category_to_item.values()) 

        # category coefficients.
        self.category_coef = nn.Parameter(
            torch.zeros(self.num_categories, self.category_feature_dim),
            requires_grad=True)
        
        # item coefficients.
        self.item_coef = nn.Parameter(
            torch.zeros(self.num_items, self.item_feature_dim),
            requires_grad=True)
        
        # needs to be (0, 1), all lambda_k init to 0.5
        self.lambdas = nn.Parameter(torch.ones(self.num_categories) / 2, requires_grad=True)
        # used to warn users if forgot to call clamp.
        self._clamp_called_flag = True

    def _check_input_shapes(self, x_category: torch.Tensor, x_item: torch.Tensor,
                            item_avilability: torch.BoolTensor) -> None:
        assert len(x_item.shape) == len(x_item.shape) == 3
        assert x_category.shape[0] == x_item.shape[0]
        T = x_category.shape[0]
        assert x_category.shape == (T, self.num_categories, self.category_feature_dim)
        assert x_item.shape == (T, self.num_items, self.item_feature_dim)
        assert item_avilability.shape == (T, self.num_items)
        
    def forwad(self,
               x_category: torch.Tensor,
               x_item: torch.Tensor,
               item_avilability: torch.BoolTensor
               ) -> torch.Tensor:
        """"Computes log P[t, i] = the log probability for the user involved in trip t to choose item i.
        Let n denote the ID of the user involved in trip t, then P[t, i] = P_{ni} on page 86 of the
        book "discrete choice methods with simulation" by Train.

        Args:
            x_category (torch.Tensor): a tensor with shape (num_trips, num_categories, *) including
                features of all categories in each trip.
            x_item (torch.Tensor): a tensor with shape (num_trips, num_items, *) including features
                of all items in each trip.
            item_avilability (torch.BoolTensor): a boolean tensor with shape (num_trips, num_items)
                indicating the aviliability of items in each trip. If item_avilability[t, i] = False,
                the utility of choosing item i in trip t, V[t, i], will be set to -inf.
                Given the decomposition V[t, i] = W[t, k(i)] + Y[t, i] + eps, V[t, i] is set to -inf
                by setting Y[t, i] = -inf for unavilable items.

        Returns:
            torch.Tensor: a tensor of shape (num_trips, num_items) including the log probabilty
            of choosing item i in trip t.
        """
        
        if not self._clamp_called_flag:
            warnings.UserWarning('Did you forget to call clamp_lambdas() after optimizer.step()?')
        # check shapes.
        self._check_shapes(x_category, x_item)

        # The overall utility of item can be decomposed into V[item] = W[category] + Y[item] + eps.
        T = x_category.shape[0]
        # compute category-specific utility.
        category_coef = self.category_coef.view(1, self.num_categories, self.category_feature_dim).expand(T, -1, -1)
        W = (x_category * category_coef).sum(dim=-1)  # (T, num_categories)
        # compute item-specific utility.
        item_coef = self.item_coef.view(1, self.num_items, self.item_feature_dim).expand(T, -1, -1)
        Y = (x_item * item_coef).sum(dim=-1)  # (T, num_items)
        # mask out unavilable items. TODO(Tianyu): is this correct?
        Y[~item_avilability] = -1.0e20

        # =============================================================================
        # compute the inclusive value of each category.
        inclusive_value = dict()
        for k, Bk in self.category_to_item.items():
            # for nest k, divide the Y of all items in Bk by lambda_k.
            Y[:, Bk] /= self.lambdas[k]
            # compute inclusive value for category k.
            inclusive_value[k] = torch.logsumexp(Y[:, Bk], dim=1, keepdim=False)  # (T,)
        
        # boardcast inclusive value from (T, num_categories) to (T, num_items).
        # for trip t, I[t, i] is the inclusive value of the category item i belongs to.
        I = torch.empty(T, self.num_items)
        # TODO(Tianyu): may need to optimize performance if num_categories is large.
        # TODO(Tianyu): check if backprop works in slice assignment.
        for k, Bk in self.category_to_item.items():
            I[:, Bk] = inclusive_value[k]  # (T,)

        # logP_item[t, i] = log P(ni|Bk), where Bk is the category item i is in, n is the user in trip t.
        logP_item = Y - I  # (T, num_items)

        # =============================================================================
        # logP_categroy[t, i] = log P(Bk), for item i in trip t, the probability of choosing the nest/bucket
        # item i belongs to. logP_category has shape (T, num_items)
        # logit[t, i] = W[n, k] + lambda[k] I[n, k], where n is the user involved in trip t, k is
        # the category item i belongs to.
        logit = torch.empty(T, self.num_items)
        for k, Bk in self.category_to_item.items():
            logit[:, Bk] = W[:, k] + self.lambdas[k] * inclusive_value[k]  # (T,)
        logP_category = logit - torch.logsumexp(logit, dim=1, keepdim=True)
        
        # =============================================================================
        # compute the joint log P_{ni} as in the textbook.
        logP = logP_item + logP_category
        self._clamp_called_flag = False
        return logP

    def evaluate(self, dataset, is_train: bool=True):
        # compute the negative log-likelihood directly.
        if is_train:
            self.train()
        else:
            self.eval()
        nll = torch.scalar_tensor(None)
        return nll

    def clamp_lambdas(self):
        """
        Restrict values of lambdas to 0 < lambda <= 1 to guarantee the utility maximization property
        of the model.
        This method should be called everytime after optimizer.step().
        We add a self_clamp_called_flag to remind researchers if this method is not called.
        """
        for k in range(len(self.lambdas)):
            self.lambdas[k] = torch.clamp(self.lambdas[k], 1e-5, 1)
        self._clam_called_flag = True

    @staticmethod
    def add_constant(x: torch.Tensor, where: str='prepend') -> torch.Tensor:
        """A helper function used to add constant to feature tensor,
        x has shape (batch_size, num_classes, num_parameters),
        returns a tensor of shape (*, num_parameters+1).
        """
        batch_size, num_classes, num_parameters = x.shape
        ones = torch.ones((batch_size, num_classes, 1))
        if where == 'prepend':
            new = torch.cat((ones, x), dim=-1)
        elif where == 'append':
            new = torch.cat((x, ones), dim=-1)
        else:
            raise Exception
        return new

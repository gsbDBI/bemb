from typing import Dict, List

import torch
import torch.nn as nn


class NestedLogitModel(nn.Module):
    # The nested logit model.
    def __init__(self,
                 num_categories: int,
                 category_to_item: Dict[object, List[int]]) -> None:
        super(NestedLogitModel, self).__init__()
        self.num_categories = num_categories
        self.categories = ...
        self.first_stage_models = self.build_first_stage_models()
        self.second_stage_models = self.build_second_stage_models()
        
        # needs to be (0, 1), all lambda_k init to 0.5
        self.lambdas = nn.Parameter(torch.ones(self.num_categories)/2, requires_grad=True)
    
    def build_first_stage_models(self):
        model_dict = dict()
        return nn.ModuleDict[model_dict]
        
    def build_second_stage_models(self):
        model_dict = dict()
        return nn.ModuleDict[model_dict]

    def forwad(self):
        p_cond_list = list()
        IV_list = list()
        
        for k in self.categories:  # TODO: can we parallelize this loop?
            # compute the second stage utility.
            features = ... 
            # T: batch size.
            # utility of items in nest k.
            Y = self.second_stage_model[k](...)  # (T, |Bk|)
            lambda_k = self.lambdas[k]  # scalar.

            # TODO: potential numercial unstability issues.
            Y_over_lambda = torch.exp(Y / lambda_k)  # (T, |Bk|)
            # compute the second stage probability.
            # probability of choosing each item i in nest Bk.
            p_cond = Y_over_lambda / Y_over_lambda.sum(dim=1).view(-1, 1)  # (T, |Bk|), boardcasted.
            p_cond_list.append(p_cond)
            # inclusive value of choosing category k.
            I_category = torch.logsumexp(Y_over_lambda, dim=1)  # (T,)
            IV_list.append(I_category)
        I = torch.stack(IV_list, dim=0)  # (T, num_categories) TODO: not sure.
        # first stage probability.
        W = self.first_stage_model(...)  # (T, num_categories)
        # I * self.lambdas.view(1, -1)  # (T, num_categories)
        nest_utility = torch.exp(W + self.lambdas.view(1, -1) * W)  # (T, num_categories)
        # prob of choose a particular nest.
        p_marginal = nest_utility / nest_utility.sum(dim=1).view(-1, 1)  # (T, num_categories)

        # prob of choosing each item P(item i, nest k) for all i in nest k.
        # TODO: change things to log scale.
        
        p_joint = torch.zeros(T, self.num_items)  # placeholder.
        for k in self.categories:
            Bk = self.category_to_item[k]
            p_joint[:, Bk] = p_marginal[k] * p_cond_list[k]  # scalar * (T, |Bk|)
        
        # predicted probability of item i.
        return p_joint

    def second_stage_forward(self):
        pass
    
    def first_stage_forward(self):
        pass

    def evaluate(self, dataset, is_train: bool=True):
        # compute the negative log-likelihood directly.
        if is_train:
            self.train()
        else:
            self.eval()
        nll = torch.scalar_tensor(None)
        return nll 

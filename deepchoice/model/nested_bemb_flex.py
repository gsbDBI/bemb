"""Draft for the BEMB model"""
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from deepchoice.model import BEMBFlex


class NestedBEMB(nn.Module):
    def __init__(self,
                 item_level_args: dict,
                 category_level_args: dict,
                 num_users: int,
                 num_items: int,
                 obs2prior_dict_item: dict,
                 obs2prior_dict_category: dict,
                 category_to_item: Dict[str, List[int]],
                 latent_dim_item: int,
                 latent_dim_category: int,
                 trace_log_q_item: Optional[bool]=False,
                 trace_log_q_category: Optional[bool]=False,
                 num_user_obs: Optional[int]=None,
                 num_item_obs: Optional[int]=None,
                 num_category_obs: Optional[int]=None,
                 num_session_obs: Optional[int]=None,
                 num_price_obs: Optional[int]=None,
                 num_taste_obs: Optional[int]=None,
                 shared_lambda: bool=False):
        self.__dict__.update(locals())
        super(NestedBEMB, self).__init__()
        # read in args.
        # self.num_users = num_users
        # self.num_items = num_items

        # self.obs2prior_dict_item = obs2prior_dict_item
        # self.obs2prior_dict_category = obs2prior_dict_category

        # self.category_to_item = category_to_item
        self.categories = list(category_to_item.keys())
        self.num_categories = len(self.categories)

        # self.latent_dim_item = latent_dim_item
        # self.latent_dim_category = latent_dim_category

        # self.trace_log_q_item = trace_log_q_item
        # self.trace_log_q_category = trace_log_q_category

        # dimension of each observable.
        # self.num_obs_dict = {
        #     'user': num_user_obs,
        #     'item': num_item_obs,
        #     'cateogry': num_category_obs,
        #     'session': num_session_obs,
        #     'price': num_price_obs,
        #     'taste': num_taste_obs
        # }
        # self.num_user_obs = num_user_obs
        # self.num_item_obs = num_item_obs
        # self.num_category_obs = num_category_obs
        # self.num_session_obs = num_session_obs
        # self.num_price_obs = num_price_obs
        # self.num_taste_obs = num_taste_obs

        # self.shared_lambda = shared_lambda

        if self.shared_lambda:
            self.lambda_weight = nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            self.lambda_weight = nn.Parameter(torch.ones(self.num_categories) / 2, requires_grad=True)

        self.BEMB_item = BEMBFlex(**item_level_args)
        self.BEMB_category = BEMBFlex(**category_level_args)

        # item maps to category in BEMB_category.
        # self.BEMB_category = BEMB(num_users=self.num_users,
        #                           num_items=self.num_categories,
        #                           obs2prior_dict=self.obs2prior_dict_category,
        #                           latent_dim=self.latent_dim_category,
        #                           trace_log_q=self.trace_log_q_category,
        #                           category_to_item=None,
        #                           num_user_obs=self.num_user_obs,
        #                           num_item_obs=self.num_category_obs,
        #                           likelihood='all')

    def forward(self, batch_item, batch_category):
        sample_dict_item = dict()
        for coef_name, variational in self.BEMB_item.variational_dict.items():
            sample_dict_item[coef_name] = variational.mean.unsqueeze(dim=0)  # (1, num_*, dim)

        sample_dict_category = dict()
        for coef_name, variational in self.BEMB_category.variational_dict.items():
            sample_dict_category[coef_name] = variational.mean.unsqueeze(dim=0)  # (1, num_*, dim)

        # there is 1 random seed in this case.
        out = self.log_likelihood(batch_item, batch_category,
                                  sample_dict_item, sample_dict_category)  # (num_seeds=1, num_sessions, num_items)
        return out.squeeze()  # (num_sessions, num_items)

    def log_likelihood(self, batch_item, batch_category,
                       sample_dict_item, sample_dict_category):
        # NOTE: batch_category.item_obs should be category level observables.
        # sample_dict_{item, category}: Monte Carlo samples for latent in BEMB_{item, category}.
        # output shape: (num_seeds, num_sessions, num_items)

        assert len(batch_item) == len(batch_category)
        num_sessions = len(batch_item)
        device = batch_item.device
        for v in sample_dict_item.values():
            num_seeds = v.shape[0]
            break

        if self.shared_lambda:
            self.lambdas = self.lambda_weight.expand(self.num_categories)
        else:
            self.lambdas = self.lambda_weight

        # category-specific utilities (W).
        # NOTE: item_obs in batch_category is category observables indeed.
        # with return_logit=True, log_likelihood returns utility instead of likelihood.
        # (num_seeds, batch_size, self.num_category)
        W = self.BEMB_category.log_likelihood(batch_category, sample_dict_category, return_logit=True)

        # item-specific utilities (Y).
        # (num_seeds, batch_size, self.num_items)
        # unavailable items are already masked out with -Inf utility given batch_item.item_availability.
        Y = self.BEMB_item.log_likelihood_all_items(batch_item, sample_dict_item, return_logit=True)

        # ==========================================================================================
        # compute the inclusive value of each category.
        # ==========================================================================================
        # TODO: vectorize this operation.
        inclusive_value = dict()
        for k, Bk in self.category_to_item.items():
            # for nest k, divide the Y of all items in Bk by lambda_k.
            # (num_seeds, num_sessions, |Bk|)
            Y[:, :, Bk] /= self.lambdas[k]
            # compute inclusive value for category k.
            # (num_seeds, num_sessions,)
            inclusive_value[k] = torch.logsumexp(Y[:, :, Bk], dim=-1, keepdim=False)

        # boardcast inclusive value from (num_seeds, num_sessions, num_categories)
        # to (num_seeds, num_sessions, num_items).
        I = torch.zeros(num_seeds, num_sessions, self.num_items).to(device)
        for k, Bk in self.category_to_item.items():
            # (num_seeds, num_sessions, len(Bk) <- (num_seeds, num_sessions, 1)
            I[:, :, Bk] = inclusive_value[k].view(num_seeds, num_sessions, 1)

        # logP_item[t, i] = log P(ni|Bk), where Bk is the category item i is in,
        # n is the user in trip t. The first dimension is for Monte Carlo seeds.
        # equivalent to log_softmax.
        logP_item = Y - I  # (num_seeds, num_sessions, num_items)

        # ==========================================================================================
        # category level probabilities.
        # ==========================================================================================

        # logP_categroy[t, i] = log P(Bk), for item i in trip t, the probability of choosing the nest/bucket
        # item i belongs to. logP_category has shape (num_sessions, num_items)
        # logit[t, i] = W[n, k] + lambda[k] I[n, k], where n is the user involved in trip t, k is
        # the category item i belongs to.
        logit = torch.zeros(num_seeds, num_sessions, self.num_items).to(device)
        for k, Bk in self.category_to_item.items():
            logit[:, :, Bk] = (W[:, :, k] + self.lambdas[k] * inclusive_value[k]).view(num_seeds, num_sessions, 1)
        # only count each category once in the logsumexp within the category level model.
        cols = [x[0] for x in self.category_to_item.values()]
        # (num_seeds, num_sessions, num_items)
        logP_category = logit - torch.logsumexp(logit[:, cols], dim=1, keepdim=True)

        # ==========================================================================================
        # compute the joint log P_{ni} as in the textbook.
        # ==========================================================================================
        # (num_seeds, num_sessions, num_items)
        logP = logP_item + logP_category
        return logP

    def elbo(self, batch_item, batch_category, num_seeds: int=1) -> torch.Tensor:
        # output shape: scalar.
        # 1. sample latent variables from their variational distributions.
        # (num_seeds, num_classes, dim)
        sample_dict_item = dict()
        for coef_name, variational in self.BEMB_item.variational_dict.items():
            sample_dict_item[coef_name] = variational.reparameterize_sample(num_seeds)

        sample_dict_category = dict()
        for coef_name, variational in self.BEMB_category.variational_dict.items():
            sample_dict_category[coef_name] = variational.reparameterize_sample(num_seeds)

        # 2. compute log p(latent) prior.
        # (num_seeds,)
        # TODO: is this correct to assume independent priors?
        log_prior = self.BEMB_item.log_prior(batch_item, sample_dict_item) \
                  + self.BEMB_category.log_prior(batch_category, sample_dict_category)
        # scalar.
        elbo = log_prior.mean()  # average over Monte Carlo samples for expectation.

        # 3. compute the log likelihood log p(obs|latent).
        # (num_seeds, num_sessions, num_items)
        log_p_all_items = self.log_likelihood(batch_item=batch_item,
                                              batch_category=batch_category,
                                              sample_dict_item=sample_dict_item,
                                              sample_dict_category=sample_dict_category)
        # (num_sessions, num_items), averaged over Monte Carlo samples for expectation at dim 0.
        log_p_all_items = log_p_all_items.mean(dim=0)

        # log_p_cond[*, session] = log prob of the item bought in this session.
        # (num_sessions,)
        log_p_chosen_items = log_p_all_items[torch.arange(len(batch_item)), batch_item.label]
        # scalar
        elbo += log_p_chosen_items.sum(dim=-1)  # sessions are independent.

        # 4. optionally add log likelihood under variational distributions q(latent).
        if self.trace_log_q_item:
            log_q_item = self.BEMB_item.log_variational(sample_dict_item)  # (num_seeds,)
            elbo -= log_q_item.mean()  # scalar.

        if self.trace_log_q_category:
            log_q_category = self.BEMB_category.log_variational(sample_dict_category)  # (num_seeds,)
            elbo -= log_q_category.mean()  # scalar.

        return elbo

    def get_within_category_accuracy(self, *args):
        return self.BEMB_item.get_within_category_accuracy(*args)

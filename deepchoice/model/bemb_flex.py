import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch_scatter import scatter_max
from torch_scatter.composite import scatter_log_softmax

from deepchoice.model.gaussian import batch_factorized_gaussian_log_prob
from deepchoice.model.bayesian_coefficient_dev import BayesianCoefficient

from torch.profiler import record_function


def parse_utility(utility_string: str) -> list:
    # split additive terms
    parameter_suffix = ('_item', '_user')
    observable_prefix = ('item_', 'user_', 'session_', 'price_', 'taste_')

    def is_parameter(name: str) -> bool:
        return any(name.endswith(suffix) for suffix in parameter_suffix)

    def is_observable(name: str) -> bool:
        return any(name.startswith(prefix) for prefix in observable_prefix)

    additive_terms = utility_string.split(' + ')
    additive_decomposition = list()
    for term in additive_terms:
        atom = {'coefficient': [], 'observable': None}
        # split multiplicative terms.
        for x in term.split(' * '):
            if is_parameter(x):
                atom['coefficient'].append(x)
            elif is_observable(x):
                atom['observable'] = x
            else:
                raise ValueError(f'{x} term cannot be classified.')
        additive_decomposition.append(atom)
    return additive_decomposition


class BEMBFlex(nn.Module):
    def __init__(self,
                 utility_formula: str,
                 num_users: int,
                 num_items: int,
                 num_sessions: int,
                 obs2prior_dict: Dict[str, bool],
                 coef_dim_dict: Dict[str, int],
                 trace_log_q: bool=False,
                 category_to_item: Dict[str, List[int]]=None,
                 likelihood: str='within_category',
                 num_user_obs: Optional[int]=None,
                 num_item_obs: Optional[int]=None,
                 num_session_obs: Optional[int]=None,
                 num_price_obs: Optional[int]=None,
                 num_taste_obs: Optional[int]=None
                 ) -> None:
        """
        Args:
            num_users (int): number of users.
            num_items (int): number of items.
            latent_dim (int): dimension of user and item latents.
            latent_dim_price (int, optional): the dimension of latents for the price coefficient.
            trace_log_q (bool, optional): whether to trace the derivative of varitional likelihood logQ
                with respect to variational parameters in the ELBO while conducting gradient update.
                Defaults to False.
            category_to_item (Dict[str, List[int]], optional): a dictionary with category id or name
                as keys, and category_to_item[C] contains the list of item ids belonging to category C.
                If None is provided, all items are assumed to be in the same category.
                Defaults to None.
            likelihood (str, optional): specifiy the method used for computing likelihood
                P(item i | user, session, ...).
                Options are
                - 'all': a softmax across all items.
                - 'within_category': firstly group items by categories and run separate softmax for each category.
                Defaults to 'within_category'.
            obs2prior_user (bool, optional): whether user observables enter the prior of user latent or not.
                Defaults to False.
            num_user_obs (Optional[int], optional): number of user observables, required only if
                obs2prior_user is True.
                Defaults to None.
            obs2prior_item (bool, optional): whether item observables enter the prior of item latent or not.
                Defaults to False.
            num_item_obs (Optional[int], optional): number of item observables, required only if
                obs2prior_item or obs2utility_item is True.
                Defaults to None.
            item_intercept (bool, optional): whether to add item-specifc intercept (lambda term) to utlity or not.
                Defaults to False.
            obs2utility_item (bool, optional): whether to allow direct effect from item observables to utility or not.
                Defaults to False.
            obs2utility_session (bool, optional): whether to allow direct effect from session observables
                to utility or not.
                Defaults to False.
            num_session_obs (Optional[int], optional): number of session observables, required only if
                obs2utility_session is True. Defaults to None.
        """
        self.__dict__.update(locals())
        super(BEMBFlex, self).__init__()

        # create a category idx tensor for faster indexing.
        if self.likelihood == 'within_category':
            self.num_categories = len(self.category_to_item)

            category_idx = torch.zeros(self.num_items)
            for c, items_in_c in self.category_to_item.items():
                category_idx[items_in_c] = c
            category_idx = category_idx.long()
        else:
            category_idx = torch.zeros(self.num_items).long()

        self.register_buffer('category_idx', category_idx)

        # ==========================================================================================
        # Create Bayesian Coefficient Objects
        # ==========================================================================================
        # model configuration.
        self.formula = parse_utility(utility_formula)
        self.raw_formula = utility_formula
        self.obs2prior_dict = obs2prior_dict

        # dimension of each observable.
        self.num_obs_dict = {
            'user': num_user_obs,
            'item': num_item_obs,
            'session': num_session_obs,
            'price': num_price_obs,
            'taste': num_taste_obs
        }

        # prior_dict = dict()
        # variational_dict = dict()
        coef_dict = dict()
        for additive_term in self.formula:
            for coef_name in additive_term['coefficient']:
                variation = coef_name.split('_')[-1]
                if variation == 'user':
                    num_classes = self.num_users
                elif variation == 'item':
                    num_classes = self.num_items
                else:
                    raise NotImplementedError

                coef_dict[coef_name] = BayesianCoefficient(variation=variation,
                                                           num_classes=num_classes,
                                                           obs2prior=self.obs2prior_dict[coef_name],
                                                           num_obs=self.num_obs_dict[variation],
                                                           dim=self.coef_dim_dict[coef_name])
        self.coef_dict = nn.ModuleDict(coef_dict)

    def __str__(self):
        return f'Bayesian EMBedding Model with U[user, item, session] = {self.raw_formula}\n' \
               + f'Total number of parameters: {self.num_params}.\n' \
               + 'With the following coefficients:\n' \
               + str(self.coef_dict)

    def forward(self, batch, return_logit: bool=False) -> torch.Tensor:
        """Computes the log likelihood of choosing each item in each session.

        Args:
            batch ([type]): [description]

        Returns:
            torch.Tensor: a tensor of shape (num_sessions, num_items) containing the log likelihood
                that each item is chosen in each session.
        """
        # TODO: need some testing.
        sample_dict = dict()
        for coef_name, coef in self.coef_dict.items():
            sample_dict[coef_name] = coef.variational_distribution.mean.unsqueeze(dim=0)  # (1, num_*, dim)

        # there is 1 random seed in this case.
        out = self.log_likelihood(batch, sample_dict, return_logit)  # (num_seeds=1, num_sessions, num_items)
        return out.squeeze()  # (num_sessions, num_items)

    @property
    def num_params(self) -> int:
        return sum([p.numel() for p in self.parameters()])

    @property
    def device(self) -> torch.device:
        for coef in self.coef_dict.values():
            return coef.device

    # ==============================================================================================
    # Helper functions.
    # ==============================================================================================

    @torch.no_grad()
    def get_within_category_accuracy(self,
                                     log_p_all_items: torch.Tensor,
                                     label: torch.LongTensor) -> Dict[str, float]:
        """A helper function for computing prediction accuracy (i.e., all non-differential metrics)
        within category.
        In particular, thie method calculates the accuracy, precision, recall and F1 score.


        This method has the same functionality as the following peusodcode:
        for C in categories:
            # get sessions in which item in category C was purchased.
            T <- (t for t in {0,1,..., len(label)-1} if label[t] is in C)
            Y <- label[T]

            predictions = list()
            for t in T:
                # get the prediction within category for this session.
                y_pred = argmax_{items in C} log prob computed before.
                predictions.append(y_pred)

            accuracy = mean(Y == predictions)

        Similarly, this function computes precision, recall and f1score as well.

        Args:
            log_p_all_items (torch.Tensor): shape (num_sessions, num_items) the log probability of
                choosing each item in each session.
            label (torch.LongTensor): shape (num_sessions,), the IDs of items purchased in each session.

        Returns:
            [Dict[str, float]]: A dictionary containing performance metrics.
        """
        # argmax: (num_sessions, num_categories), within category argmax.
        # item IDs are consecutive, thus argmax is the same as IDs of the item with highest P.
        # self.category_idx = self.category_idx.to(self.device)
        _, argmax_by_category = scatter_max(log_p_all_items, self.category_idx, dim=-1)

        # category_purchased[t] = the category of item label[t].
        # (num_sessions,)
        category_purchased = self.category_idx[label]

        # pred[t] = the item with highest utility from the category item label[t] belongs to.
        # (num_sessions,)
        pred_from_category = argmax_by_category[torch.arange(len(label)), category_purchased]

        within_category_accuracy = (pred_from_category == label).float().mean().item()

        # precision
        precision = list()

        recall = list()
        for i in range(self.num_items):
            correct_i = torch.sum((torch.logical_and(pred_from_category == i, label == i)).float())
            precision_i = correct_i / torch.sum((pred_from_category == i).float())
            recall_i = correct_i / torch.sum((label == i).float())

            # do not add if divided by zero.
            if torch.any(pred_from_category == i):
                precision.append(precision_i.cpu().item())
            if torch.any(label == i):
                recall.append(recall_i.cpu().item())

        precision = float(np.mean(precision))
        recall = float(np.mean(recall))

        if precision == recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {'accuracy': within_category_accuracy,
                'precision': precision,
                'recall': recall,
                'f1score': f1}

    # ==============================================================================================
    # Methods for terms in the ELBO: prior, likelihood, and variational.
    # ==============================================================================================
    def log_likelihood(self,
                       batch,
                       sample_dict,
                       return_logit: bool=False
                       ) -> torch.Tensor:
        """Computes the log probability of choosing each item in each session based on current model
        parameters.
        This method allows for specifying {user, item}_latent_value for Monte Carlo estimation in ELBO.
        For actual prediction tasks, use the forward() function, which will use means of varitional
        distributions for user and item latents.

        Args:
            batch (ChoiceDataset): a ChoiceDataset object containing relevant information.
            sample_dict(Dict[str, torch.Tensor]): Monte Carlo samples for model coefficients
                (i.e., those Greek letters).
                sample_dict.keys() should be the same as keys of self.obs2prior_dict, i.e., those
                greek letters actually enter the functional form of utility.
                The value of sample_dict should be tensors of shape (num_seeds, num_classes, dim)
                where num_classes in {num_users, num_items, 1}
                and dim in {latent_dim(K), num_item_obs, num_user_obs, 1}.

        Returns:
            torch.Tensor: a tensor of shape (num_seeds, num_sessions, self.num_items), where
                out[x, y, z] is the proabbility of choosing item z in session y conditioned on user
                and item latents to be the x-th Monte Carlo sample.
        """
        assert hasattr(batch, 'user_index')
        assert sample_dict.keys() == self.obs2prior_dict.keys()

        # get the base utility of each item for each user with shape (num_seeds, num_users, num_items).

        for v in sample_dict.values():
            num_seeds = v.shape[0]
            break

        # utility = torch.zeros(num_seeds, self.num_users, self.num_items).to(self.device)

        class PositiveInteger(object):
            def __eq__(self, other):
                return isinstance(other, int) and other > 0

        positive_integer = PositiveInteger()

        # ==========================================================================================
        # Direct Construction.
        # ==========================================================================================
        # short-hands for easier shape check.
        R = num_seeds
        P = len(batch)  # num_purchases.
        S = self.num_sessions
        U = self.num_users
        I = self.num_items

        def reshape_item_coef_sample(C):
            # input shape (R, I, *)
            C = C.view(R, 1, I, -1).expand(-1, P, -1, -1)
            assert C.shape == (R, P, I, positive_integer)
            return C

        def reshape_user_coef_sample(C):
            # input shape (R, U, *)
            C = C.view(R, U, 1, -1).expand(-1, -1, I, -1)  # (R, U, I, *)
            C = C[:, batch.user_index, :, :]
            assert C.shape == (R, P, I, positive_integer)
            return C

        def reshape_coef_sample(sample, name):
            if name.endswith('_user'):
                return reshape_user_coef_sample(sample)
            elif name.endswith('_item'):
                return reshape_item_coef_sample(sample)
            else:
                raise ValueError

        def reshape_observable(obs, name):
            O = obs.shape[-1]  # numberof observables.
            assert O == positive_integer
            if name.startswith('item_'):
                assert obs.shape == (I, O)
                obs = obs.view(1, 1, I, O).expand(R, P, -1, -1)
            elif name.startswith('user_'):
                assert obs.shape == (U, O)
                obs = obs[batch.user_index, :]  # (P, O)
                obs = obs.view(1, P, 1, O).expand(R, -1, I, -1)
            elif name.startswith('session_'):
                assert obs.shape == (S, O)
                obs = obs[batch.session_index, :]  # (P, O)
                return obs.view(1, P, 1, O).expand(R, -1, I, -1)
            elif name.startswith('price_'):
                assert obs.shape == (S, I, O)
                obs = obs[batch.session_index, :, :]  # (P, I, O)
                return obs.view(1, P, I, O).expand(R, -1, -1, -1)
            elif name.startswith('taste_'):
                assert obs.shape == (U, I, O)
                obs = obs[batch.user_index, :, :]  # (P, I, O)
                return obs.view(1, P, I, O).expand(R, -1, -1, -1)
            else:
                raise ValueError
            assert obs.shape == (R, P, I, O)
            return obs

        # (random_seeds, num_purchases, num_items).
        # utility = torch.zeros(R, P, I).to(self.device)
        # utility = torch.Tensor([[[0]]]).to(self.device)
        # utility = torch.zeros([1, 1, 1]).to(self.device)

        # loop over additive term to utility
        additive_term_list = list()
        for term in self.formula:
            if len(term['coefficient']) == 0 and term['observable'] is None:
                raise ValueError

            elif len(term['coefficient']) == 1 and term['observable'] is None:
                # E.g., lambda_item or lambda_user
                coef_name = term['coefficient'][0]
                coef_sample = reshape_coef_sample(sample_dict[coef_name], coef_name)
                assert coef_sample.shape == (R, P, I, 1)
                additive_term = coef_sample.view(R, P, I)

            elif len(term['coefficient']) == 2 and term['observable'] is None:
                # E.g., <theta_user, lambda_item>.
                coef_name_0 = term['coefficient'][0]
                coef_name_1 = term['coefficient'][1]

                coef_sample_0 = reshape_coef_sample(sample_dict[coef_name_0], coef_name_0)
                coef_sample_1 = reshape_coef_sample(sample_dict[coef_name_1], coef_name_1)

                assert coef_sample_0.shape == coef_sample_1.shape == (R, P, I, positive_integer)

                additive_term = (coef_sample_0 * coef_sample_1).sum(dim=-1)

            elif len(term['coefficient']) == 1 and term['observable'] is not None:
                # E.g., theta_user * x_obs_item
                coef_name = term['coefficient'][0]
                coef_sample = reshape_coef_sample(sample_dict[coef_name], coef_name)
                assert coef_sample.shape == (R, P, I, PositiveInteger())

                obs_name = term['observable']
                obs = reshape_observable(getattr(batch, obs_name), obs_name)
                assert obs.shape == (R, P, I, PositiveInteger())

                additive_term = (coef_sample * obs).sum(dim=-1)

            elif len(term['coefficient']) == 2 and term['observable'] is not None:
                coef_name_0 = term['coefficient'][0]
                coef_name_1 = term['coefficient'][1]

                coef_sample_0 = reshape_coef_sample(sample_dict[coef_name_0], coef_name_0)
                coef_sample_1 = reshape_coef_sample(sample_dict[coef_name_1], coef_name_1)
                assert coef_sample_0.shape == coef_sample_1.shape == (R, P, I, positive_integer)
                O_times_latent_dim = coef_sample_0.shape[-1]

                obs_name = term['observable']
                obs = reshape_observable(getattr(batch, obs_name), obs_name)
                assert obs.shape == (R, P, I, positive_integer)
                O = obs.shape[-1]  # number of observables.

                assert (O_times_latent_dim % O) == 0
                latent_dim = O_times_latent_dim // O

                coef_sample_0 = coef_sample_0.view(R, P, I, O, latent_dim)
                coef_sample_1 = coef_sample_1.view(R, P, I, O, latent_dim)
                # compute the coefficient with shape (R, P, I, O).
                coef = (coef_sample_0 * coef_sample_1).sum(dim=-1)

                additive_term = (coef * obs).sum(dim=-1)

            else:
                raise ValueError
            additive_term_list.append(additive_term)

        utility = torch.stack(additive_term_list).sum(dim=0)

        if batch.item_availability is not None:
            # expand to the Monte Carlo sample dimension.
            assert batch.item_availability[batch.session_index, :].shape == (P, I)
            A = batch.item_availability[batch.session_index, :].unsqueeze(dim=0).view(R, P, I)
            utility[~A] = -1.0e20

        if return_logit:
            # output shape: (num_seeds, num_sessions, self.num_items)
            return utility

        # compute log likelihood log p(choosing item i | user, item latents)
        if self.likelihood == 'all':
            # compute log softmax for all items all together.
            log_p = log_softmax(utility, dim=-1)
        elif self.likelihood == 'within_category':
            # compute log softmax separately within each category.
            # log_p = scatter_log_softmax(utility, self.category_idx.to(self.device), dim=-1)
            log_p = scatter_log_softmax(utility, self.category_idx, dim=-1)
        # output shape: (num_seeds, num_sessions, self.num_items)
        return log_p

    def log_prior(self, batch, sample_dict):
        for sample in sample_dict.values():
            num_seeds = sample.shape[0]
            break

        # total = torch.zeros(num_seeds).to(self.device)
        log_prior_list = list()
        for coef_name, coef in self.coef_dict.items():
            # log_prob outputs (num_seeds, num_{items, users}), sum to (num_seeds).

            if self.obs2prior_dict[coef_name]:
                if coef_name.endswith('_item'):
                    x_obs = batch.item_obs
                elif coef_name.endswith('_user'):
                    x_obs = batch.user_obs
                else:
                    raise ValueError
            else:
                x_obs = None

            log_prior_list.append(coef.log_prior(sample=sample_dict[coef_name], x_obs=x_obs).sum(dim=-1))
        total = torch.stack(log_prior_list).sum(dim=0)
        return total

    def log_variational(self, sample_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.Tensor([0]).to(self.device)

        log_variational_list = list()

        for coef_name, coef in self.coef_dict.items():
            # log_prob outputs (num_seeds, num_{items, users}), sum to (num_seeds).
            # total += coef.log_variational(sample_dict[coef_name]).sum(dim=-1)
            log_variational_list.append(coef.log_variational(sample_dict[coef_name]).sum(dim=-1))

        total = torch.stack(log_variational_list).sum(dim=0)
        return total

    def elbo(self, batch, num_seeds: int=1) -> torch.Tensor:
        """Computes the current ELBO.

        Args:
            batch (ChoiceDataset): a ChoiceDataset containing necessary infromation.
            num_seeds (int, optional): the number of Monte Carlo samples from variational distributions
                to evaluate the expectation in ELBO.
                Defaults to 1.

        Returns:
            torch.Tensor: a scalar tensor of the ELBO estimated from num_seeds Monte Carlo samples.
        """
        # 1. sample latent variables from their variational distributions.
        # (num_seeds, num_classes, dim)
        sample_dict = dict()
        for coef_name, coef in self.coef_dict.items():
            sample_dict[coef_name] = coef.reparameterize_sample(num_seeds)

        # 2. compute log p(latent) prior.
        # (num_seeds,)
        log_prior = self.log_prior(batch, sample_dict)
        # scalar
        elbo = log_prior.mean()  # average over Monte Carlo samples for expectation.

        # 3. compute the log likelihood log p(obs|latent).
        # (num_seeds, num_sessions, num_items)
        log_p_all_items = self.log_likelihood(batch, sample_dict)
        # (num_sessions, num_items), averaged over Monte Carlo samples for expectation at dim 0.
        log_p_all_items = log_p_all_items.mean(dim=0)

        # log_p_cond[*, session] = log prob of the item bought in this session.
        # (num_sessions,)
        log_p_chosen_items = log_p_all_items[torch.arange(len(batch)), batch.label]
        # scalar
        elbo += log_p_chosen_items.sum(dim=-1)  # sessions are independent.

        # 4. optionally add log likelihood under variational distributions q(latent).
        if self.trace_log_q:
            log_q = self.log_variational(sample_dict)  # (num_seeds,)
            elbo -= log_q.mean()  # scalar.

        return elbo

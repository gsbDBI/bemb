"""
The Bayesian EMBedding (BEMB) model.
A futher attempt to speed things up, split the (1) training step, only calculate utilitis for items
in categories bought during training (2) compute all utilities during the inference time.
"""
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from deepchoice.model.bayesian_coefficient import BayesianCoefficient
from termcolor import cprint
from torch_scatter import scatter_max
from torch_scatter.composite import scatter_log_softmax


class PositiveInteger(object):
    def __eq__(self, other):
        return isinstance(other, int) and other > 0


def parse_utility(utility_string: str) -> list:
    # split additive terms
    parameter_suffix = ('_item', '_user', '_constant')
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
                 obs2prior_dict: Dict[str, bool],
                 coef_dim_dict: Dict[str, int],
                 num_items: int,
                 num_users: Optional[int] = None,
                 num_sessions: Optional[int] = None,
                 trace_log_q: bool = False,
                 category_to_item: Dict[str, List[int]] = None,
                 likelihood: str = 'within_category',
                 num_user_obs: Optional[int] = None,
                 num_item_obs: Optional[int] = None,
                 num_session_obs: Optional[int] = None,
                 num_price_obs: Optional[int] = None,
                 num_taste_obs: Optional[int] = None
                 ) -> None:
        """
        Args:
            utility_formula (str): a string representing the utility function U[user, item, session].
                See documentation for more details in the documentation for the format of formula.
                Examples:
                    lambda_item
                    lambda_item + theta_user * alpha_item + zeta_user * item_obs
                    lambda_item + theta_user * alpha_item + gamma_user * beta_item * price_obs

            obs2prior_dict (Dict[str, bool]): a dictionary maps coefficient name (e.g., 'lambda_item')
                to a boolean indicating if observable (e.g., item_obs) enters the prior of the coefficient.

            coef_dim_dict (Dict[str, int]): a dictionary maps coefficient name (e.g., 'lambda_item')
                to an integer indicating the dimension of coefficeint.
                For standalone coefficients like U = lamdba_item, the dim should be 1.
                For factorized coefficients like U = theta_user * alpha_item, the dim should be the
                    latent dimension of theta and alpha.
                For coefficients multiplied with observables like U = zeta_user * item_obs, the dim
                    should be the number of observables in item_obs.
                For factorized coefficient muplied with observables like U = gamma_user * beta_item * price_obs,
                    the dim should be the latent dim multiplied by number of observables in price_obs.

            num_items (int): number of items.
            num_users (int, optional): number of users, required only if coefficient or observable
                depending on user is in utitliy. Defaults to None.
            num_sessions (int, optional): number of sessions, required only if coefficient or
                observable depending on session is in utility. Defaults to None.

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

            num_{user, item, session, price, taste}_obs (int, optional): number of observables of
                each type of features, only required if observable enters prior.
                NOTE: currently we only allow coefficient to depend on either user or item, thus only
                user and item observables can enter the prior of coefficient. Hence session, price,
                and taste observables are never required, we include it here for completeness.
        """
        super(BEMBFlex, self).__init__()
        cprint('You are using BEMB Flex v3', 'red')
        self.utility_formula = utility_formula
        self.obs2prior_dict = obs2prior_dict
        self.coef_dim_dict = coef_dim_dict
        self.num_items = num_items
        self.num_users = num_users
        self.num_sessions = num_sessions
        self.trace_log_q = trace_log_q
        self.category_to_item = category_to_item
        self.likelihood = likelihood
        assert self.likelihood == 'within_category'  # TODO: remove this.

        # ==========================================================================================
        # Category ID to Item ID mapping.
        # Category ID to Category Size mapping.
        # Item ID to Category ID mapping.
        # ==========================================================================================
        self.num_categories = len(self.category_to_item)

        max_category_size = max(len(x) for x in category_to_item.values())
        category_to_item_tensor = torch.full((self.num_categories, max_category_size), -1)
        category_to_size_tensor = torch.empty(self.num_categories)

        for c, item_in_c in self.category_to_item.items():
            category_to_item_tensor[c, :len(item_in_c)] = torch.LongTensor(item_in_c)
            category_to_size_tensor[c] = torch.scalar_tensor(len(item_in_c))

        self.register_buffer('category_to_item_tensor', category_to_item_tensor.long())
        self.register_buffer('category_to_size_tensor', category_to_size_tensor.long())

        item_to_category_tensor = torch.zeros(self.num_items)
        for c, items_in_c in self.category_to_item.items():
            item_to_category_tensor[items_in_c] = c
        self.register_buffer('item_to_category_tensor', item_to_category_tensor.long())

        # ==========================================================================================
        # Create Bayesian Coefficient Objects
        # ==========================================================================================
        # model configuration.
        self.formula = parse_utility(utility_formula)
        print('utility formula parsed:\n', self.formula)
        self.raw_formula = utility_formula
        self.obs2prior_dict = obs2prior_dict

        # dimension of each observable.
        self.num_obs_dict = {
            'user': num_user_obs,
            'item': num_item_obs,
            'session': num_session_obs,
            'price': num_price_obs,
            'taste': num_taste_obs,
            'constant': 1
        }

        variation_to_num_classes = {
            'user': self.num_users,
            'item': self.num_items,
            'constant': 1
        }

        coef_dict = dict()
        for additive_term in self.formula:
            for coef_name in additive_term['coefficient']:
                variation = coef_name.split('_')[-1]
                coef_dict[coef_name] = BayesianCoefficient(variation=variation,
                                                           num_classes=variation_to_num_classes[variation],
                                                           obs2prior=self.obs2prior_dict[coef_name],
                                                           num_obs=self.num_obs_dict[variation],
                                                           dim=self.coef_dim_dict[coef_name])
        self.coef_dict = nn.ModuleDict(coef_dict)

    def __str__(self):
        return f'Bayesian EMBedding Model with U[user, item, session] = {self.raw_formula}\n' \
               + f'Total number of parameters: {self.num_params}.\n' \
               + 'With the following coefficients:\n' \
               + str(self.coef_dict)

    def forward(self, batch, return_logit: bool = False, all_items: bool = True) -> torch.Tensor:
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
            sample_dict[coef_name] = coef.variational_distribution.mean.unsqueeze(
                dim=0)  # (1, num_*, dim)

        # there is 1 random seed in this case.
        # (num_seeds=1, num_sessions, num_items)
        if all_items:
            out = self.log_likelihood_all_items(batch, sample_dict, return_logit)
        else:
            out = self.log_likelihood(batch, sample_dict, return_logit)
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
        _, argmax_by_category = scatter_max(
            log_p_all_items, self.item_to_category_tensor, dim=-1)

        # category_purchased[t] = the category of item label[t].
        # (num_sessions,)
        category_purchased = self.item_to_category_tensor[label]

        # pred[t] = the item with highest utility from the category item label[t] belongs to.
        # (num_sessions,)
        pred_from_category = argmax_by_category[torch.arange(
            len(label)), category_purchased]

        within_category_accuracy = (
            pred_from_category == label).float().mean().item()

        # precision
        precision = list()

        recall = list()
        for i in range(self.num_items):
            correct_i = torch.sum(
                (torch.logical_and(pred_from_category == i, label == i)).float())
            precision_i = correct_i / \
                torch.sum((pred_from_category == i).float())
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
    def log_likelihood_all_items(self,
                                 batch,
                                 sample_dict: Dict[str, torch.Tensor],
                                 return_logit: bool = False
                                 ) -> torch.Tensor:
        """
        NOTE: this method computes utilities for all items available.

        Computes the log probability of choosing each item in each session based on current model
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
        # assert we have sample for all coefficients.
        assert sample_dict.keys() == self.coef_dict.keys()
        for v in sample_dict.values():
            num_seeds = v.shape[0]
            break

        user_session_index = torch.stack([batch.user_index, batch.session_index])
        assert user_session_index.shape == (2, len(batch))
        unique_user_sess, inverse_indices = torch.unique(user_session_index, dim=1, return_inverse=True)

        user_index = unique_user_sess[0, :]
        session_index = unique_user_sess[1, :]

        positive_integer = PositiveInteger()

        # ==========================================================================================
        # Compute the utility tensor U with shape (random_seeds, num_purchases, num_items).
        # where U[seed_id, p, i] indicates the utility for the user in the p-th purchase record to
        # purchase item i according to the seed_id-th random sample of coefficient.
        # ==========================================================================================
        # short-hands for easier shape check.
        R = num_seeds
        # P = len(batch)  # num_purchases.
        P = unique_user_sess.shape[1]
        S = self.num_sessions
        U = self.num_users
        I = self.num_items
        # ==========================================================================================
        # Helper Functions for Reshaping.
        # ==========================================================================================
        def reshape_user_coef_sample(C):
            # input shape (R, U, *)
            C = C.view(R, U, 1, -1).expand(-1, -1, I, -1)  # (R, U, I, *)
            C = C[:, user_index, :, :]
            assert C.shape == (R, P, I, positive_integer)
            return C

        def reshape_item_coef_sample(C):
            # input shape (R, I, *)
            C = C.view(R, 1, I, -1).expand(-1, P, -1, -1)
            assert C.shape == (R, P, I, positive_integer)
            return C

        def reshape_constant_coef_sample(C):
            # input shape (R, *)
            C = C.view(R, 1, 1, -1).expand(-1, P, I, -1)
            assert C.shape == (R, P, I, positive_integer)
            return C

        def reshape_coef_sample(sample, name):
            # reshape the monte carlo sample of coefficients to (R, P, I, *).
            if name.endswith('_user'):
                # (R, U, *) --> (R, P, I, *)
                return reshape_user_coef_sample(sample)
            elif name.endswith('_item'):
                # (R, I, *) --> (R, P, I, *)
                return reshape_item_coef_sample(sample)
            elif name.endswith('_constant'):
                # (R, *) --> (R, P, I, *)
                return reshape_constant_coef_sample(sample)
            else:
                raise ValueError

        def reshape_observable(obs, name):
            # reshape observable to (R, P, I, *) so that it can be multiplied with monte carlo
            # samples of coefficients.
            O = obs.shape[-1]  # numberof observables.
            assert O == positive_integer
            if name.startswith('item_'):
                assert obs.shape == (I, O)
                obs = obs.view(1, 1, I, O).expand(R, P, -1, -1)
            elif name.startswith('user_'):
                assert obs.shape == (U, O)
                obs = obs[user_index, :]  # (P, O)
                obs = obs.view(1, P, 1, O).expand(R, -1, I, -1)
            elif name.startswith('session_'):
                assert obs.shape == (S, O)
                obs = obs[session_index, :]  # (P, O)
                return obs.view(1, P, 1, O).expand(R, -1, I, -1)
            elif name.startswith('price_'):
                assert obs.shape == (S, I, O)
                obs = obs[session_index, :, :]  # (P, I, O)
                return obs.view(1, P, I, O).expand(R, -1, -1, -1)
            elif name.startswith('taste_'):
                assert obs.shape == (U, I, O)
                obs = obs[user_index, :, :]  # (P, I, O)
                return obs.view(1, P, I, O).expand(R, -1, -1, -1)
            else:
                raise ValueError
            assert obs.shape == (R, P, I, O)
            return obs

        # ==========================================================================================
        # Compute Components contigent to users and items only.
        # ==========================================================================================
        utility = torch.zeros(R, U, I, device=self.device)

        # user and item interactions.
        def is_time_invariant_observable(name):
            return name is None or name.endswith('_item') or name.endswith('_user')

        def is_time_invariant_term(term):
            return is_time_invariant_observable(term['observable'])

        def _reshape(sample, name):
            if name.endswith('_constant'):
                out = sample.view(R, 1, 1, -1).expand(-1, U, I, -1)
            elif name.endswith('_item'):
                out = sample.view(R, 1, I, -1).expand(-1, U, -1, -1)
            elif name.endswith('_user'):
                out = sample.view(R, U, 1, -1).expand(-1, -1, I, -1)
            return out

        for term in self.formula:
            if is_time_invariant_term(term):
                if len(term['coefficient']) == 1 and term['observable'] is None:
                    out = _reshape(sample_dict[term['coefficient'][0]], term['coefficient'][0])
                    utility += out.view(R, U, I)

                elif len(term['coefficient']) == 2 and term['observable'] is None:
                    A = _reshape(sample_dict[term['coefficient'][0]], term['coefficient'][0])
                    B = _reshape(sample_dict[term['coefficient'][1]], term['coefficient'][1])
                    utility += (A * B).sum(dim=-1).view(R, U, I)

                elif len(term['coefficient']) == 1 and term['observable'] is not None:
                    coef = _reshape(sample_dict[term['coefficient'][0]], term['coefficient'[0]])
                    obs_name = term['observable']
                    if obs_name.startswith('item_'):
                        obs = getattr(batch, obs_name).view(1, 1, I, -1).expand(R, U, -1, -1)
                    elif obs_name.startswith('user_'):
                        obs = getattr(batch, obs_name).view(1, U, 1, -1).expand(R, -1, I, -1)
                    utility += (coef * obs).sum(dim=-1).view(R, U, I)

        # ==========================================================================================
        # Copmute the Utility Term by Term.
        # ==========================================================================================
        # (random_seeds, num_purchases, num_items).
        # utility = torch.zeros(R, P, I, device=self.device)
        utility = utility[:, user_index, :]

        # loop over additive term to utility
        for term in self.formula:
            if not is_time_invariant_term(term):
                pass

            # Type I: single coefficient, e.g., lambda_item or lambda_user.
            if len(term['coefficient']) == 1 and term['observable'] is None:
                # E.g., lambda_item or lambda_user
                coef_name = term['coefficient'][0]
                coef_sample = reshape_coef_sample(sample_dict[coef_name], coef_name)
                assert coef_sample.shape == (R, P, I, 1)
                additive_term = coef_sample.view(R, P, I)

            # Type II: factorized coefficient, e.g., <theta_user, lambda_item>.
            elif len(term['coefficient']) == 2 and term['observable'] is None:
                coef_name_0 = term['coefficient'][0]
                coef_name_1 = term['coefficient'][1]

                coef_sample_0 = reshape_coef_sample(sample_dict[coef_name_0], coef_name_0)
                coef_sample_1 = reshape_coef_sample(sample_dict[coef_name_1], coef_name_1)

                assert coef_sample_0.shape == coef_sample_1.shape == (R, P, I, positive_integer)

                additive_term = (coef_sample_0 * coef_sample_1).sum(dim=-1)

            # Type III: single coefficient multiplied by observable, e.g., theta_user * x_obs_item.
            elif len(term['coefficient']) == 1 and term['observable'] is not None:
                coef_name = term['coefficient'][0]
                coef_sample = reshape_coef_sample(sample_dict[coef_name], coef_name)
                assert coef_sample.shape == (R, P, I, positive_integer)

                obs_name = term['observable']
                obs = reshape_observable(getattr(batch, obs_name), obs_name)
                assert obs.shape == (R, P, I, positive_integer)

                additive_term = (coef_sample * obs).sum(dim=-1)

            # Type IV: factorized coefficient multiplied by observable.
            # e.g., gamma_user * beta_item * price_obs.
            elif len(term['coefficient']) == 2 and term['observable'] is not None:
                coef_name_0, coef_name_1 = term['coefficient'][0], term['coefficient'][1]

                coef_sample_0 = reshape_coef_sample(sample_dict[coef_name_0], coef_name_0)
                coef_sample_1 = reshape_coef_sample(sample_dict[coef_name_1], coef_name_1)
                assert coef_sample_0.shape == coef_sample_1.shape == (R, P, I, positive_integer)
                num_obs_times_latent_dim = coef_sample_0.shape[-1]

                obs_name = term['observable']
                obs = reshape_observable(getattr(batch, obs_name), obs_name)
                assert obs.shape == (R, P, I, positive_integer)
                num_obs = obs.shape[-1]  # number of observables.

                assert (num_obs_times_latent_dim % num_obs) == 0
                latent_dim = num_obs_times_latent_dim // num_obs

                coef_sample_0 = coef_sample_0.view(R, P, I, num_obs, latent_dim)
                coef_sample_1 = coef_sample_1.view(R, P, I, num_obs, latent_dim)
                # compute the factorized coefficient with shape (R, P, I, O).
                coef = (coef_sample_0 * coef_sample_1).sum(dim=-1)

                additive_term = (coef * obs).sum(dim=-1)

            else:
                raise ValueError(f'Undefined term type: {term}')

            assert additive_term.shape == (R, P, I)
            utility += additive_term

        # ==========================================================================================
        # Mask Out Unavailable Items in Each Session.
        # ==========================================================================================

        if batch.item_availability is not None:
            # expand to the Monte Carlo sample dimension.
            # (S, I) -> (P, I) -> (1, P, I) -> (R, P, I)
            A = batch.item_availability[session_index, :].unsqueeze(dim=0).expand(R, -1, -1)
            utility[~A] = -1.0e20

        utility = utility[:, inverse_indices, :]
        assert utility.shape == (R, len(batch), I)

        if return_logit:
            # output shape: (num_seeds, num_purchases, num_items)
            return utility

        # compute log likelihood log p(choosing item i | user, item latents)
        # compute log softmax separately within each category.
        log_p = scatter_log_softmax(utility, self.item_to_category_tensor, dim=-1)
        # output shape: (num_seeds, num_purchases, num_items)
        return log_p

    def log_likelihood(self,
                       batch,
                       sample_dict: Dict[str, torch.Tensor],
                       return_logit: bool = False
                       ) -> torch.Tensor:
        """
        NOTE: this method is more efficient.

        Computes the log probability of choosing each item in each session based on current model
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
        # assert we have sample for all coefficients.
        assert sample_dict.keys() == self.coef_dict.keys()
        for v in sample_dict.values():
            num_seeds = v.shape[0]
            break

        # get category id of each item bought.
        cate_index = self.item_to_category_tensor[batch.label]

        # get item ids of all items from the same category of each item bought.
        relevant_item_index = self.category_to_item_tensor[cate_index, :]
        relevant_item_index = relevant_item_index.view(-1,)
        relevant_item_index = relevant_item_index[relevant_item_index != -1]

        # the first repeats[0] entries in relevant_item_index are for the category of label[0]
        repeats = self.category_to_size_tensor[cate_index]
        # argwhere(reverse_indices == k) are positions in relevant_item_indexs for the category of label[k].
        reverse_indices = torch.repeat_interleave(torch.arange(len(batch), device=self.device), repeats)
        # expand the user_index and session_index.
        user_index = torch.repeat_interleave(batch.user_index, repeats)
        session_index = torch.repeat_interleave(batch.session_index, repeats)
        label_expanded = torch.repeat_interleave(batch.label, repeats)

        # use for shape check.
        positive_integer = PositiveInteger()

        # ==========================================================================================
        # Compute the utility tensor U with shape (random_seeds, num_purchases, num_items).
        # where U[seed_id, p, i] indicates the utility for the user in the p-th purchase record to
        # purchase item i according to the seed_id-th random sample of coefficient.
        # ==========================================================================================
        # short-hands for easier shape check.
        R = num_seeds
        total_computation = len(session_index)  # total number of relevant items.
        S = self.num_sessions
        U = self.num_users
        I = self.num_items
        # ==========================================================================================
        # Helper Functions for Reshaping.
        # ==========================================================================================

        def reshape_coef_sample(sample, name):
            # reshape the monte carlo sample of coefficients to (R, P, I, *).
            if name.endswith('_user'):
                # (R, U, *) --> (R, total_computation, *)
                return sample[:, user_index, :]
            elif name.endswith('_item'):
                # (R, I, *) --> (R, total_computation, *)
                return sample[:, relevant_item_index, :]
            elif name.endswith('_constant'):
                # (R, *) --> (R, total_computation, *)
                return sample.view(R, 1, -1).expand(-1, total_computation, -1)
            else:
                raise ValueError

        def reshape_observable(obs, name):
            # reshape observable to (R, P, I, *) so that it can be multiplied with monte carlo
            # samples of coefficients.
            O = obs.shape[-1]  # numberof observables.
            assert O == positive_integer
            if name.startswith('item_'):
                assert obs.shape == (I, O)
                obs = obs[relevant_item_index, :]
            elif name.startswith('user_'):
                assert obs.shape == (U, O)
                obs = obs[user_index, :]
            elif name.startswith('session_'):
                assert obs.shape == (S, O)
                obs = obs[session_index, :]
            elif name.startswith('price_'):
                assert obs.shape == (S, I, O)
                obs = obs[session_index, relevant_item_index, :]
            elif name.startswith('taste_'):
                assert obs.shape == (U, I, O)
                obs = obs[user_index, relevant_item_index, :]
            else:
                raise ValueError
            assert obs.shape == (total_computation, O)
            return obs.unsqueeze(dim=0).expand(R, -1, -1)

        # ==========================================================================================
        # Compute Components contigent to users and items only.
        # ==========================================================================================
        utility = torch.zeros(R, total_computation, device=self.device)

        # loop over additive term to utility
        for term in self.formula:
            # Type I: single coefficient, e.g., lambda_item or lambda_user.
            if len(term['coefficient']) == 1 and term['observable'] is None:
                # E.g., lambda_item or lambda_user
                coef_name = term['coefficient'][0]
                coef_sample = reshape_coef_sample(sample_dict[coef_name], coef_name)
                assert coef_sample.shape == (R, total_computation, 1)
                additive_term = coef_sample.view(R, total_computation)

            # Type II: factorized coefficient, e.g., <theta_user, lambda_item>.
            elif len(term['coefficient']) == 2 and term['observable'] is None:
                coef_name_0 = term['coefficient'][0]
                coef_name_1 = term['coefficient'][1]

                coef_sample_0 = reshape_coef_sample(sample_dict[coef_name_0], coef_name_0)
                coef_sample_1 = reshape_coef_sample(sample_dict[coef_name_1], coef_name_1)

                assert coef_sample_0.shape == coef_sample_1.shape == (R, total_computation, positive_integer)

                additive_term = (coef_sample_0 * coef_sample_1).sum(dim=-1)

            # Type III: single coefficient multiplied by observable, e.g., theta_user * x_obs_item.
            elif len(term['coefficient']) == 1 and term['observable'] is not None:
                coef_name = term['coefficient'][0]
                coef_sample = reshape_coef_sample(sample_dict[coef_name], coef_name)
                assert coef_sample.shape == (R, total_computation, positive_integer)

                obs_name = term['observable']
                obs = reshape_observable(getattr(batch, obs_name), obs_name)
                assert obs.shape == (R, total_computation, positive_integer)

                additive_term = (coef_sample * obs).sum(dim=-1)

            # Type IV: factorized coefficient multiplied by observable.
            # e.g., gamma_user * beta_item * price_obs.
            elif len(term['coefficient']) == 2 and term['observable'] is not None:
                coef_name_0, coef_name_1 = term['coefficient'][0], term['coefficient'][1]
                coef_sample_0 = reshape_coef_sample(sample_dict[coef_name_0], coef_name_0)
                coef_sample_1 = reshape_coef_sample(sample_dict[coef_name_1], coef_name_1)
                assert coef_sample_0.shape == coef_sample_1.shape == (R, total_computation, positive_integer)
                num_obs_times_latent_dim = coef_sample_0.shape[-1]

                obs_name = term['observable']
                obs = reshape_observable(getattr(batch, obs_name), obs_name)
                assert obs.shape == (R, total_computation, positive_integer)
                num_obs = obs.shape[-1]  # number of observables.

                assert (num_obs_times_latent_dim % num_obs) == 0
                latent_dim = num_obs_times_latent_dim // num_obs

                coef_sample_0 = coef_sample_0.view(R, total_computation, num_obs, latent_dim)
                coef_sample_1 = coef_sample_1.view(R, total_computation, num_obs, latent_dim)
                # compute the factorized coefficient with shape (R, P, I, O).
                coef = (coef_sample_0 * coef_sample_1).sum(dim=-1)

                additive_term = (coef * obs).sum(dim=-1)

            else:
                raise ValueError(f'Undefined term type: {term}')

            assert additive_term.shape == (R, total_computation)
            utility += additive_term

        # ==========================================================================================
        # Mask Out Unavailable Items in Each Session.
        # ==========================================================================================

        if batch.item_availability is not None:
            # expand to the Monte Carlo sample dimension.
            A = batch.item_availability[session_index, relevant_item_index].unsqueeze(dim=0).expand(R, -1)
            utility[~A] = -1.0e20

        # compute log likelihood log p(choosing item i | user, item latents)
        if not return_logit:
            # compute the log probability from logits/utilities.
            log_p = scatter_log_softmax(utility, reverse_indices, dim=-1)
        # TODO: check if this makes sense.
        log_p = log_p[:, label_expanded == relevant_item_index]
        # output shape: (num_seeds, num_purchases, num_items)
        return log_p

    def log_prior(self, batch, sample_dict: Dict[str, torch.Tensor]) -> torch.scalar_tensor:
        """Calculates the log-likelihood of Monte Carlo samples of Bayesian coefficients under their
        prior distribution. This method assume coefficients are statistically independnet.

        Args:
            batch ([type]): a dataset object contains observables for computing the prior distribution
                if obs2prior is True.
            sample_dict (Dict[str, torch.Tensor]): a dictionary coefficient names to Monte Carlo
                samples.

        Raises:
            ValueError: [description]

        Returns:
            torch.scalar_tensor: [description]
        """
        assert sample_dict.keys() == self.coef_dict.keys()
        for sample in sample_dict.values():
            num_seeds = sample.shape[0]
            break

        total = torch.zeros(num_seeds, device=self.device)

        for coef_name, coef in self.coef_dict.items():
            if self.obs2prior_dict[coef_name]:
                if coef_name.endswith('_item'):
                    x_obs = batch.item_obs
                elif coef_name.endswith('_user'):
                    x_obs = batch.user_obs
                else:
                    raise ValueError(f'No observable found to support obs2prior for {coef_name}.')
            else:
                x_obs = None

            # log_prob outputs (num_seeds, num_{items, users}), sum to (num_seeds).
            total += coef.log_prior(sample=sample_dict[coef_name], x_obs=x_obs).sum(dim=-1)
        return total

    def log_variational(self, sample_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate the log-likelihood of smaples in sample_dict under the current variational
        distribution.

        Args:
            sample_dict (Dict[str, torch.Tensor]):  a dictionary coefficient names to Monte Carlo
                samples.

        Returns:
            torch.Tensor: [description]
        """
        total = torch.zeros(1, device=self.device)

        for coef_name, coef in self.coef_dict.items():
            # log_prob outputs (num_seeds, num_{items, users}), sum to (num_seeds).
            total += coef.log_variational(sample_dict[coef_name]).sum(dim=-1)

        return total

    def elbo(self, batch, num_seeds: int = 1) -> torch.Tensor:
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
        elbo = self.log_prior(batch, sample_dict).mean(dim=0)  # (num_seeds,) -> scalar.

        # 3. compute the log likelihood log p(obs|latent).
        # sum over independent purchase decision for individual items, mean over MC seeds.
        # (num_sessions, num_relevant_items) -> scalar.
        elbo += self.log_likelihood(batch, sample_dict).sum(dim=1).mean(dim=0)

        # 4. optionally add log likelihood under variational distributions q(latent).
        if self.trace_log_q:
            elbo -= self.log_variational(sample_dict).mean(dim=0)

        return elbo
"""Draft for the BEMB model"""
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch_scatter import scatter_max
from torch_scatter.composite import scatter_log_softmax

from gaussian import batch_factorized_gaussian_log_prob


class VariationalFactorizedGaussian(nn.Module):
    """A helper class initializes a batch of factorized (i.e., Gaussian distribution with diagional
    standard covariance matrix) Gaussian distributions.
    This class is used as the variational family for real-valued latent variables.
    """
    def __init__(self, num_classes: int, dim: int) -> None:
        """

        Args:
            num_classes (int): the number of Gaussian distributions to create. For example, if we
                want the variational distribution of each user's latent to be a 10-dimensional Gaussian,
                then num_classes is set to the number of users. The same holds while we are creating
                variational distribution for item latent variables.
            dim (int): the dimension of each Gaussian distribution. In above example, dim is set to 10.
        """
        super(VariationalFactorizedGaussian, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.mean = nn.Parameter(torch.zeros(num_classes, dim), requires_grad=True)
        self.logstd = nn.Parameter(torch.ones(num_classes, dim), requires_grad=True)

    def __repr__(self) -> str:
        return f'VariationalFactorizedGaussian(num_classes={self.num_classes}, dim_out={self.dim})'

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """For each batch B and class C, computes the log probability of value[B, C, :] under the
            C-th Gaussian distribution. See the doc string for `batch_factorized_gaussian_log_prob`
            for more details.

        Args:
            value (torch.Tensor): a tensor with shape (batch_size, num_classes, dim_out).

        Returns:
            torch.Tensor: a tensor with shape (batch_size, num_classes).
        """
        return batch_factorized_gaussian_log_prob(self.mean, self.logstd, value)

    def reparameterize_sample(self, num_seeds: int=1) -> torch.Tensor:
        """Samples from the multivariate Gaussian distribution using the reparameterization trick.

        Args:
            num_seeds (int): number of samples generated.

        Returns:
            torch.Tensor: a tensor of shape (num_seeds, num_classes, dim), where out[:, C, :] follows
                the C-th Gaussian distribution.
        """
        # create random seeds from N(0, 1).
        eps = torch.randn(num_seeds, self.num_classes, self.dim).to(self.device)
        # parameters for each Gaussian distribution, boardcast across random seeds.
        mu = self.mean.view(1, self.num_classes, self.dim)
        std = torch.exp(self.logstd).view(1, self.num_classes, self.dim)
        out = mu + std * eps
        assert out.shape == (num_seeds, self.num_classes, self.dim)
        return out


class LearnableGaussianPrior(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, std: Union[str, float, torch.Tensor]=1.0) -> None:
        """Construct a Gaussian distribution for prior of user/item embeddigns, whose mean and
        standard deviation depends on user/item observables.
        NOTE: to avoid exploding number of parameters, learnable parameters in this class are shared
            across all items/users.
        NOTE: we have not supported the standard deviation to be dependent on observables yet.

        For example:
        p(alpha_ik | H_k, obsItem_i) = Gaussian( mean=H_k*obsItem_i, variance=s2obsPrior )
        p(beta_ik | H'_k, obsItem_i) = Gaussian( mean=H'_k*obsItem_i, variance=s2obsPrior )

        Args:
            dim_in (int): the number of input features.
            dim_out (int): the dimension of latent features.
            std (Union[str, float]): the standard deviation of latent features.
                Options are
                0. float: a pre-specified constant standard deviation for all dimensions of Gaussian.
                1. a tensor with length dim_out with pre-specified constant standard devation.
                2. 'learnable_scalar': a learnable standard deviation shared across all dimensions of Gaussian.
                3. 'learnable_vector': use a separate learnable standard deviation for each dimension of Gaussian.
        """
        super(LearnableGaussianPrior, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.H = nn.Linear(dim_in, dim_out, bias=False)

        if isinstance(std, float):
            self.logstd = torch.log(torch.scalar_tensor(std)).expand(self.dim_out)
        elif isinstance(std, torch.Tensor):
            self.logstd = torch.log(std)
        elif std == 'learnable_scalar':
            # TODO(Tianyu): check if this expand function works as we expected. Check the number of parameters.
            self.logstd = nn.Parameter(torch.zeros(1), requires_grad=True).expand(self.dim_out)
        elif std == 'learnable_vector':
            self.logstd = nn.Parameter(torch.zeros(self.dim_out), requires_grad=True)
        else:
            raise ValueError(f'Unsupported standard deviation option {std}')

    def __repr__(self) -> str:
        return f'LearnableGaussianPrior(dim_in={self.dim_in}, dim_out={self.dim_out})'

    def log_prob(self, x_obs: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Compute the log likelihood of `value` given observables `x_obs`.

        Args:
            x_obs (torch.Tensor): a tensor with shape (num_classes, dim_in) such as item observbales
                or user observables, where num_classes is corresponding to the number of items or
                number of users.
            value (torch.Tensor): a tensor of shape (batch_size, num_classes, dim_out).

        Returns:
            torch.Tensor: output shape (batch_size, num_classes)
        """
        # compute mean vector for each class.
        mu = self.H(x_obs)  # (num_classes, self.dim_out)
        # expand standard deviations shared across all classes.
        logstd = self.logstd.unsqueeze(dim=0).expand(x_obs.shape[0], -1).to(x_obs.device)  # (num_classes, self.dim_out)
        return batch_factorized_gaussian_log_prob(mu, logstd, value)


class StandardGaussianPrior(nn.Module):
    """A helper class for evaluating the log_prob of Monte Carlo samples for latent variables on
    a N(0, 1) prior.
    """
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super(StandardGaussianPrior, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

    def __repr__(self) -> str:
        return f'StandardGaussianPrior(dim_out={self.dim_out})'

    def log_prob(self, x_obs: object, value: torch.Tensor) -> torch.Tensor:
        """Compute the log-likelihood of `value` under N(0, 1)

        Args:
            x_obs (object): x_obs is not used at all, it's here to make args of log_prob consistent
                with LearnableGaussianPrior.log_prob().
            value (torch.Tensor): a tensor of shape (batch_size, num_classes, dim_out).

        Returns:
            torch.Tensor: output shape (batch_size, num_classes)
        """
        batch_size, num_classes, dim_out = value.shape
        assert dim_out == self.dim_out
        mu = torch.zeros(num_classes, self.dim_out).to(value.device)
        logstd = torch.zeros(num_classes, self.dim_out).to(value.device)  # (num_classes, self.dim_out)
        out = batch_factorized_gaussian_log_prob(mu, logstd, value)
        assert out.shape == (batch_size, num_classes)
        return out


class BEMB(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 obs2prior_dict: Dict[str, bool],
                 latent_dim: int,
                 latent_dim_price: Optional[int]=None,
                 trace_log_q: bool=False,
                 category_to_item: Dict[str, List[int]]=None,
                 likelihood: str='within_category',
                 num_user_obs: Optional[int]=None,
                 num_item_obs: Optional[int]=None,
                 num_session_obs: Optional[int]=None,
                 num_price_obs: Optional[int]=None,
                 num_taste_obs: Optional[int]=None  # Not used for now.
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
            likelihood (str, optional): specifiy the method used for computing likelihood P(item i | user, session, ...).
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
            obs2utility_session (bool, optional): whether to allow direct effect from session observables to utility or not.
                Defaults to False.
            num_session_obs (Optional[int], optional): number of session observables, required only if
                obs2utility_session is True. Defaults to None.
        """
        super(BEMB, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.latent_dim_price = latent_dim_price
        self.trace_log_q = trace_log_q
        self.category_to_item = category_to_item
        self.likelihood = likelihood

        self.num_user_obs = num_user_obs
        self.num_item_obs = num_item_obs
        self.num_session_obs = num_session_obs
        self.num_price_obs = num_price_obs
        self.num_taste_obs = num_taste_obs

        # create a category idx tensor for faster indexing.
        if self.likelihood == 'within_category':
            self.num_categories = len(self.category_to_item)

            category_idx = torch.zeros(self.num_items)
            for c, items_in_c in self.category_to_item.items():
                category_idx[items_in_c] = c
            self.category_idx = category_idx.long()

        # ==========================================================================================
        # Create Prior Distributions.
        # ==========================================================================================
        # model configuration.
        self.obs2prior_dict = obs2prior_dict

        # dimension of each observable.
        self.num_obs_dict = {
            'user': num_user_obs,
            'item': num_item_obs,
            'session': num_session_obs,
            'price': num_price_obs,
            'taste': num_taste_obs
        }

        # dimension of each latent variable.
        has_price = (self.latent_dim_price is not None) and (self.num_price_obs is not None)
        self.coef_dim_dict = {
            'lambda_item': 1,
            'theta_user': self.latent_dim,
            'alpha_item': self.latent_dim,
            'zeta_user': self.num_item_obs,
            'iota_item': self.num_user_obs,
            'delta_item': None,  # TODO: update this.
            'gamma_user': self.latent_dim_price * self.num_price_obs if has_price else None,
            'beta_item': self.latent_dim_price * self.num_price_obs if has_price else None
        }

        prior_dict = dict()
        for coef_name, obs2prior in self.obs2prior_dict.items():
            subject = coef_name.split('_')[-1]  # {user, item, session, price, taste}
            if obs2prior:
                prior_dict[coef_name] = LearnableGaussianPrior(dim_in=self.num_obs_dict[subject],
                                                               dim_out=self.coef_dim_dict[coef_name])
            else:
                prior_dict[coef_name] = StandardGaussianPrior(dim_in=self.num_obs_dict[subject],
                                                              dim_out=self.coef_dim_dict[coef_name])
        self.prior_dict = nn.ModuleDict(prior_dict)

        # ==========================================================================================
        # Create Variational Distributions.
        # ==========================================================================================
        variational_dict = dict()
        # for now, assume all variational distributions are factorized Gaussians.
        for coef_name in self.obs2prior_dict.keys():
            if coef_name.endswith('_user'):
                num_classes = self.num_users
            elif coef_name.endswith('_item'):
                num_classes = self.num_items
            else:
                num_classes = 1  # learnable constant.

            variational_dict[coef_name] = VariationalFactorizedGaussian(num_classes,
                                                                        self.coef_dim_dict[coef_name])

        self.variational_dict = nn.ModuleDict(variational_dict)

        # self._validate_args()

    def forward(self, batch) -> torch.Tensor:
        """Computes the log likelihood of choosing each item in each session.

        Args:
            batch ([type]): [description]

        Returns:
            torch.Tensor: a tensor of shape (num_sessions, num_items) containing the log likelihood
                that each item is chosen in each session.
        """
        # TODO: need some testing.
        sample_dict = dict()
        for coef_name, variational in self.variational_dict.items():
            sample_dict[coef_name] = variational.mean.unsqueeze(dim=0)  # (1, num_*, dim)

        # there is 1 random seed in this case.
        out = self.log_likelihood(batch, sample_dict)  # (num_seeds=1, num_sessions, num_items)
        return out.squeeze()  # (num_sessions, num_items)

    # def _validate_args(self):
    #     # TODO: add more meaningful error message.
    #     assert self.likelihood in ['all', 'within_category']

    #     if self.obs2prior_user:
    #         assert self.num_user_obs is not None

    #     if self.obs2prior_item or self.obs2utility_item:
    #         assert self.num_item_obs is not None

    #     if self.obs2utility_session:
    #         assert self.num_session_obs is not None

    @property
    def num_params(self) -> int:
        return sum([p.numel() for p in self.parameters()])

    @property
    def device(self) -> torch.device:
        for prior in self.variational_dict.values():
            return prior.device

    # ==============================================================================================
    # Helper functions.
    # ==============================================================================================

    @torch.no_grad()
    def get_within_category_accuracy(self, log_p_all_items: torch.Tensor, label: torch.LongTensor) -> float:
        """A helper function for computing prediction accuracy within category.
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

        Args:
            log_p_all_items (torch.Tensor): shape (num_sessions, num_items) the log probability of
                choosing each item in each session.
            label (torch.LongTensor): shape (num_sessions,), the IDs of items purchased in each session.

        Returns:
            [float]: A float of within category accuracy computed from the above pesudo-code.
        """
        # argmax: (num_sessions, num_categories), within category argmax.
        # item IDs are consecutive, thus argmax is the same as IDs of the item with highest P.
        self.category_idx = self.category_idx.to(self.device)
        _, argmax_by_category = scatter_max(log_p_all_items, self.category_idx, dim=-1)

        # category_purchased[t] = the category of item label[t].
        # (num_sessions,)
        category_purchased = self.category_idx[label]

        # pred[t] = the item with highest utility from the category item label[t] belongs to.
        # (num_sessions,)
        pred_from_category = argmax_by_category[torch.arange(len(label)), category_purchased]

        within_category_accuracy = (pred_from_category == label).float().mean().item()
        return within_category_accuracy

    # ==============================================================================================
    # Methods for terms in the ELBO: prior, likelihood, and variational.
    # ==============================================================================================

    def log_likelihood(self,
                       batch,
                       sample_dict,
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
        assert hasattr(batch, 'user_onehot') and hasattr(batch, 'label')
        assert sample_dict.keys() == self.obs2prior_dict.keys()

        # get the base utility of each item for each user with shape (num_seeds, num_users, num_items).

        for v in sample_dict.values():
            num_seeds = v.shape[0]
            break

        utility = torch.zeros(num_seeds, self.num_users, self.num_items).to(self.device)

        # ==========================================================================================
        # 1. Time-invariant part of utility.
        # ==========================================================================================

        # 1.a. intercept term, lambda_item.
        if 'lambda_item' in sample_dict.keys():
            assert sample_dict['lambda_item'].shape == (num_seeds, self.num_items, 1)
            utility += torch.transpose(sample_dict['lambda_item'], 1, 2)  # boardcast across users.

        # 1.b. theta_user and alpha_item interaction term.
        if 'theta_user' in sample_dict.keys() and 'alpha_item' in sample_dict.keys():
            assert sample_dict['theta_user'].shape == (num_seeds, self.num_users, self.latent_dim)
            assert sample_dict['alpha_item'].shape == (num_seeds, self.num_items, self.latent_dim)
            # (num_seeds, num_users, latent_dim) bmm (num_seeds, latent_dim, num_items) -> (num_seeds, num_users, num_items)
            utility += torch.bmm(sample_dict['theta_user'], torch.transpose(sample_dict['alpha_item'], 1, 2))

        # 1.c. zeta_user * x_item_obs
        if 'zeta_user' in sample_dict.keys():
            assert sample_dict['zeta_user'].shape == (num_seeds, self.num_users, self.num_item_obs)
            assert batch.item_obs.shape == (self.num_items, self.num_item_obs)

            # TODO: double check this.
            item_obs = batch.item_obs.view(1, 1, self.items, self.num_item_obs)
            zeta = sample_dict['zeta_user'].view(num_seeds, self.num_users, 1, self.num_item_obs)
            out = (item_obs * zeta).sum(dim=1)
            assert out.shape == (num_seeds, self.num_users, self.num_items)
            utility += out

        # 1.d. iota_item * x_user_obs
        if 'iota_item' in sample_dict.keys():
            assert sample_dict['iota_item'].shape == (num_seeds, self.num_items, self.num_user_obs)
            assert batch.user_obs.shape == (self.num_users, self.num_user_obs)

            user_obs = batch.user_obs.view(1, self.num_users, 1, self.num_user_obs)
            iota = sample_dict['iota_item'].view(num_seeds, 1, self.num_items, self.num_user_obs)
            out = (user_obs * iota).sum(dim=-1)
            assert out.shape == (num_seeds, self.num_users, self.num_user)
            utility += out

        # ==========================================================================================
        # 2. Time-variant part of utility.
        # ==========================================================================================
        # convert to utility by session now.
        # get the utility for choosing each items by the user corresponding to that session.
        # get the index of the user who was making decision in each session.
        user_idx = torch.nonzero(batch.user_onehot, as_tuple=True)[1].to(self.device)
        num_sessions = user_idx.shape[0]
        utility_by_session = utility[:, user_idx, :]  # (num_seeds, num_sessions, num_items)

        # 2.a. mu_i * delta_w
        if 'mu_item' in sample_dict.keys():
            raise NotImplementedError

        # 2.b. weekday_id.
        # TODO:

        # 2.c. price variable.
        if 'gamma_user' in sample_dict.keys() and 'beta_item' in sample_dict.keys():
            # change dim to self.latent_dim * self.num_price_obs.
            assert sample_dict['gamma_user'].shape == (num_seeds, self.num_users, self.latent_dim_price * self.num_price_obs)
            assert sample_dict['beta_item'].shape == (num_seeds, self.num_items, self.latent_dim_price * self.num_price_obs)
            # support single price for now.
            assert batch.price_obs.shape == (num_sessions, self.num_items, self.num_price_obs)

            gamma_user = sample_dict['gamma_user'].view(num_seeds, self.num_users, 1, self.num_price_obs, self.latent_dim_price)
            beta_item = sample_dict['beta_item'].view(num_seeds, 1, self.num_items, self.num_price_obs, self.latent_dim_price)

            coef = (gamma_user * beta_item).sum(dim=-1)
            assert coef.shape(num_seeds, self.num_users, self.num_items, self.num_price_obs)
            coef = coef[:, user_idx, :, :]
            assert coef.shape == (num_seeds, num_sessions, self.num_items, self.num_price_obs)
            price_obs = batch.price_obs.view(1, num_sessions, self.num_items, self.num_price_obs)
            out = (coef * price_obs).sum(dim=-1)
            assert out.shape == (num_seeds, num_sessions, self.num_items)
            utility_by_session += out
            # TODO: warn reseachers to create -log(price) in the dataloader!

        assert utility_by_session.shape == (num_seeds, num_sessions, self.num_items)

        # compute log likelihood log p(choosing item i | user, item latents)
        if self.likelihood == 'all':
            # compute log softmax for all items all together.
            log_p = log_softmax(utility_by_session, dim=-1)
        elif self.likelihood == 'within_category':
            # compute log softmax separately within each category.
            log_p = scatter_log_softmax(utility_by_session, self.category_idx.to(self.device), dim=-1)
        # output shape: (num_seeds, num_sessions, self.num_items)
        return log_p

    def log_prior(self, batch, sample_dict):
        for sample in sample_dict.values():
            num_seeds = sample.shape[0]
            break

        total = torch.zeros(num_seeds).to(self.device)
        for coef_name, prior in self.prior_dict.items():
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
            total += prior.log_prob(x_obs, sample_dict[coef_name]).sum(dim=-1)
        return total

    def log_variational(self, sample_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.scalar_tensor(0.0).to(self.device)
        for coef_name, variational in self.variation_dict.items():
            # log_prob outputs (num_seeds, num_{items, users}), sum to (num_seeds).
            total += variational.log_prob(sample_dict[coef_name]).sum(dim=-1)
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
        for coef_name, variational in self.variational_dict.items():
            sample_dict[coef_name] = variational.reparameterize_sample(num_seeds)

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

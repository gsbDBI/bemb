"""Draft for the BEMB model"""
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from deepchoice.data import ChoiceDataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import log_softmax
from torch_scatter import scatter_max
from torch_scatter.composite import scatter_log_softmax

from gaussian import batch_factorized_gaussian_log_prob

# TODO(Tianyu): change attribute names like log_p --> log_prob, likelihood etc.


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

    def log_prob(self, x_obs: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Compute the log likelihood

        Args:
            x_obs (torch.Tensor): a tensor with shape (num_classes, dim_in) such as item observbales
                or user observables, where num_classes is corresponding to the number of items or
                number of users.
            value (torch.Tensor): a tensor of shape (batch_size, num_classes, dim_out).

        Returns:
            torch.Tensor: output shape (batch_size, num_classes)
        """
        # compute
        mu = self.H(x_obs)  # (num_classes, self.dim_out)
        # expand standard deviations shared across all classes.
        logstd = self.logstd.unsqueeze(dim=0).expand(x_obs.shape[0], -1).to(x_obs.device)  # (num_classes, self.dim_out)
        return batch_factorized_gaussian_log_prob(mu, logstd, value)


class BEMB(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 latent_dim: int,
                 trace_log_q: bool=False,
                 category_to_item: Dict[str, List[int]]=None,
                 likelihood: str='within_category',
                 obs2prior_user: bool=False,
                 num_user_features: Optional[int]=None,
                 obs2prior_item: bool=False,
                 num_item_features: Optional[int]=None,
                 item_intercept: bool=False,
                 obs2utility_item: bool=False,
                 obs2utility_session: bool=False,
                 num_session_features: Optional[int]=None,
                 ) -> None:
        """

        Args:
            num_users (int): number of users.
            num_items (int): number of items.
            latent_dim (int): dimension of user and item latents.
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
            num_user_features (Optional[int], optional): number of user observables, required only if
                obs2prior_user is True.
                Defaults to None.
            obs2prior_item (bool, optional): whether item observables enter the prior of item latent or not.
                Defaults to False.
            num_item_features (Optional[int], optional): number of item observables, required only if
                obs2prior_item or obs2utility_item is True.
                Defaults to None.
            item_intercept (bool, optional): whether to add item-specifc intercept (lambda term) to utlity or not.
                Defaults to False.
            obs2utility_item (bool, optional): whether to allow direct effect from item observables to utility or not.
                Defaults to False.
            obs2utility_session (bool, optional): whether to allow direct effect from session observables to utility or not.
                Defaults to False.
            num_session_features (Optional[int], optional): number of session observables, required only if
                obs2utility_session is True. Defaults to None.
        """
        super(BEMB, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.trace_log_q = trace_log_q
        self.category_to_item = category_to_item
        self.likelihood = likelihood
        self.obs2prior_user = obs2prior_user
        self.num_user_features = num_user_features
        self.obs2prior_item = obs2prior_item
        self.num_item_features = num_item_features
        self.item_intercept = item_intercept
        self.obs2utility_item = obs2utility_item
        self.obs2utility_session = obs2utility_session
        self.num_session_features = num_session_features

        # create a category idx tensor for faster indexing.
        if self.likelihood == 'within_category':
            self.num_categories = len(self.category_to_item)

            category_idx = torch.zeros(self.num_items)
            for c, items_in_c in self.category_to_item.items():
                category_idx[items_in_c] = c
            self.category_idx = category_idx.long()

        # Q: build variational distributions for user and item latents.
        self.user_latent_q = VariationalFactorizedGaussian(num_users, latent_dim)
        self.item_latent_q = VariationalFactorizedGaussian(num_items, latent_dim)

        # P: construct learnable priors for latents, if observables enter prior of latent.
        if self.obs2prior_user:
            self.user_latent_prior = LearnableGaussianPrior(dim_in=num_user_features, dim_out=latent_dim)

        if self.obs2prior_item:
            self.item_latent_prior = LearnableGaussianPrior(dim_in=num_item_features, dim_out=latent_dim)

        if self.item_intercept:
            self.item_intercept_layer = nn.Parameter(torch.zeros(self.num_items), requires_grad=True)

        # TODO(Tianyu): do we allow for user obs to affect utility as well?

        if self.obs2utility_item:
            self.obs2utility_item_layer = nn.Linear(num_item_features, 1, bias=False)

        if self.obs2utility_session:
            self.obs2utility_session_layer = nn.Linear(num_session_features, 1, bias=False)

        # an internal tracker for for tracking performance across batches.
        self.running_performance_dict = {'accuracy': [],
                                         'log_likelihood': []}

        self._validate_args()

    def forward(self, batch) -> torch.Tensor:
        """Computes the log likelihood of choosing each item in each session.

        Args:
            batch ([type]): [description]

        Returns:
            torch.Tensor: a tensor of shape (num_sessions, num_items) containing the log likelihood
                that each item is chosen in each session.
        """
        # TODO: need some testing.
        user_latent = self.user_latent_q.mean.unsqueeze(dim=0)  # (1, num_users, latent_dim)
        item_latent = self.item_latent_q.mean.unsqueeze(dim=0)  # (1, num_items, latent_dim)
        # there is 1 random seed in this case.
        out = self.log_likelihood(batch,
                                  user_latnet_value=user_latent,
                                  item_latent_value=item_latent)  # (num_seeds=1, num_sessions, num_items)
        return out.squeeze()  # (num_sessions, num_items)

    def _validate_args(self):
        # TODO: add more meaningful error message.
        assert self.likelihood in ['all', 'within_category']

        if self.obs2prior_user:
            assert self.num_user_features is not None

        if self.obs2prior_item or self.obs2utility_item:
            assert self.num_item_features is not None

        if self.obs2utility_session:
            assert self.num_session_features is not None

    @property
    def num_params(self) -> int:
        return sum([p.numel() for p in self.parameters()])

    @property
    def device(self) -> torch.device:
        return self.user_latent_q.device

    # ==============================================================================================
    # Helper functions.
    # ==============================================================================================

    # def load_params(self, path: str) -> None:
    #     """A helper function loads learned parameters from BEMB cpp to the current model.

    #     Args:
    #         path (str): the output path of BEMB cpp.
    #     """
    #     def load_cpp_tsv(file):
    #         df = pd.read_csv(os.path.join(path, file), sep='\t', index_col=0, header=None)
    #         return torch.Tensor(df.values[:, 1:])

    #     cpp_theta_mean = load_cpp_tsv('param_theta_mean.tsv')
    #     cpp_theta_std = load_cpp_tsv('param_theta_std.tsv')

    #     cpp_alpha_mean = load_cpp_tsv('param_alpha_mean.tsv')
    #     cpp_alpha_std = load_cpp_tsv('param_alpha_std.tsv')

    #     # theta user
    #     self.user_latent_q.mean = cpp_theta_mean.to(self.device)
    #     self.user_latent_q.logstd = torch.log(cpp_theta_std).to(self.device)
    #     # alpha item
    #     self.item_latent_q.mean = cpp_alpha_mean.to(self.device)
    #     self.item_latent_q.logstd = torch.log(cpp_alpha_std).to(self.device)

    def report_performance(self) -> dict:
        """Reports content in the internal performance tracker, this method empties trackers after reporting."""
        report = dict()
        for k, v in self.running_performance_dict.items():
            report[k] = np.mean(v)
            # reset tracker.
            self.running_performance_dict[k] = []
        return report

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
    # Additive terms on utility function.
    # Given certain observables, these functions compute additive terms with shape (num_sessions, num_items)
    # to utility from choosing each item in each session.
    # Type signatures of all methods in this section are the same.
    # All functions named with additive_* should return a tensor of shape (num_sessions, num_items).
    # ==============================================================================================

    def additive_intercept(self, batch) -> torch.Tensor:
        if self.item_intercept:
            return self.item_intercept_layer.view(1, batch.num_items).expand(batch.num_sessions, -1)
        else:
            return torch.zeros(batch.num_sessions, batch.num_items).to(self.device)

    def additive_obs2utility_item(self, batch) -> torch.Tensor:
        if self.obs2utility_item:
            # Add obs2utility_item here.
            item_feat = batch.item_features  # (num_items, num_item_features)
            o2u = self.obs2utility_item_layer(item_feat)   # (num_items, 1)
            return o2u.view(1, batch.num_items).expand(batch.num_sessions, -1)
        else:
            return torch.zeros(batch.num_sessions, batch.num_items).to(self.device)

    def additive_obs2utility_session(self, batch) -> torch.Tensor:
        if self.obs2utility_session:
            raise NotImplementedError
        else:
            return torch.zeros(batch.num_sessions, batch.num_items).to(self.device)

    # ==============================================================================================
    # Methods for terms in the ELBO: prior, likelihood, and variational.
    # ==============================================================================================

    def log_likelihood(self,
                       batch,
                       user_latent_value: torch.Tensor,
                       item_latent_value: torch.Tensor
                       ) -> torch.Tensor:
        """Computes the log probability of choosing each item in each session based on current model
        parameters.
        This method allows for specifying {user, item}_latent_value for Monte Carlo estimation in ELBO.
        For actual prediction tasks, use the forward() function, which will use means of varitional
        distributions for user and item latents.

        Args:
            batch (ChoiceDataset): a ChoiceDataset object containing relevant information.
            user_latent_value (torch.Tensor): a tensor of shape (num_seeds, num_users, latent_dim),
                where the first dimension denotes Monte Carlo samples from the variational distribution
                of user latent.
            item_latent_value (torch.Tesnor): a tensor of shape (num_seeds, num_items, latent_dim)
                where the first dimension denotes Monte Carlo samples from the variational distribution
                of item latent.

        Returns:
            torch.Tensor: a tensor of shape (num_seeds, num_sessions, self.num_items), where
                out[x, y, z] is the proabbility of choosing item z in session y conditioned on user
                and item latents to be the x-th Monte Carlo sample.
        """
        assert hasattr(batch, 'user_onehot') and hasattr(batch, 'label')
        # get the index of the user who was making decision in each session.
        user_idx = torch.nonzero(batch.user_onehot, as_tuple=True)[1].to(self.device)
        # get the base utility of each item for each user.
        # transpose item_latent_value (num_seeds, num_items, latent_dim) -> (num_seeds, latent_dim, num_items).
        # (num_seeds, num_users, latent_dim) bmm (num_seeds, latent_dim, num_items) -> (num_seeds, num_users, num_items)
        utility = torch.bmm(user_latent_value, torch.transpose(item_latent_value, 1, 2))
        # get the utility for choosing each items by the user corresponding to that session.
        utility_by_session = utility[:, user_idx, :]  # (num_seeds, num_sessions, num_items)

        # add additive terms to the utility function.
        # all additive_* functions returns (num_sessions, num_items) tensor,
        # unsqueeze them to (1, num_sessions, num_items) to for boardcasting across multiple random seeds.
        # item-specific intercept term.
        utility_by_session += self.additive_intercept(batch).unsqueeze(dim=0)
        # direct impact of item observables to utility function.
        utility_by_session += self.additive_obs2utility_item(batch).unsqueeze(dim=0)
        # direct impact of session observables to utility function.
        utility_by_session += self.additive_obs2utility_session(batch).unsqueeze(dim=0)

        # compute log likelihood log p(choosing item i | user, item latents)
        if self.likelihood == 'all':
            # compute log softmax for all items all together.
            log_p = log_softmax(utility_by_session, dim=-1)
        elif self.likelihood == 'within_category':
            # compute log softmax separately within each category.
            log_p = scatter_log_softmax(utility_by_session, self.category_idx.to(self.device), dim=-1)
        # output shape: (num_seeds, num_sessions, self.num_items)
        return log_p

    def log_prior(self,
                  user_latent_sample: torch.Tensor,
                  item_latent_sample: torch.Tensor,
                  user_features: Optional[torch.Tensor]=None,
                  item_features: Optional[torch.Tensor]=None
                  ) -> torch.Tensor:
        """Computes the log probability of Monte Carlo samples of user/item latents under their prior
            distributions. This method assumes the first dimension of {user, item}_latent_sample
            to be the index of Monte Carlo samples.

        Args:
            user_latent_sample (torch.Tensor): a tensor with shape (num_seeds, num_users, emb_dim)
                oof Monte Carlo samples for user latent variables.
            item_latent_sample (torch.Tensor): a tensor with shape (num_seeds, num_items, emb_dim)
                of Monte Carlo samples for item latent variables.
            user_features (Optional[torch.Tensor], optional): a tensor with shape (num_users, num_user_features)
                of user features, which may enter the prior of user latent.
                Defaults to None.
            item_features (Optional[torch.Tensor], optional): a tensor with shape (num_items, num_item_features)
                of item features, which may enter the prior of item latent.
                Defaults to None.

        Returns:
            torch.Tensor: a tensor of shape (num_seeds,), where out[i] is the (joint) log prior of
                the i-th Monte Carlo sample of user and item sample.
        """
        if self.obs2prior_user:
            log_prob_user = self.user_latent_prior.log_prob(x_obs=user_features, value=user_latent_sample)
        else:
            standard_gaussian = MultivariateNormal(loc=torch.zeros(self.latent_dim).to(self.device),
                                                   covariance_matrix=torch.eye(self.latent_dim).to(self.device))
            log_prob_user = standard_gaussian.log_prob(user_latent_sample)
        # (num_seeds, num_users)

        if self.obs2prior_item:
            log_prob_item = self.item_latent_prior.log_prob(x_obs=item_features, value=item_latent_sample)
        else:
            standard_gaussian = MultivariateNormal(loc=torch.zeros(self.latent_dim).to(self.device),
                                                   covariance_matrix=torch.eye(self.latent_dim).to(self.device))
            log_prob_item = standard_gaussian.log_prob(item_latent_sample)
        # (num_seeds, num_items)

        # sum across different users and items, prior of latent assumes latent of different user/item
        # are independent.
        return log_prob_user.sum(dim=-1) + log_prob_item.sum(dim=-1)

    def log_variational(self,
                        user_latent_sample: torch.Tensor,
                        item_latent_sample: torch.Tensor
                        ) -> torch.Tensor:
        """Compute the log likelihood of Monte Carlo samples of user and item latents under the
            current variational distributions for them.

        Args:
            user_latent_sample (torch.Tensor): A tesor of shape (num_seeds, num_users, latent_dim) of
                Monte Carlo samples of user latents.
            item_latent_sample (torch.Tensor): A tesor of shape (num_seeds, num_item, latent_dim) of
                Monte Carlo samples of item latents.

        Returns:
            torch.Tensor: a tensor of shape (num_seeds,), where out[i] is the (joint) log likelihood of
                the i-th Monte Carlo sample of user and item sample under current variational distributions.
        """
        log_q_user = self.user_latent_q.log_prob(user_latent_sample)  # (num_seeds, num_users)
        log_q_item = self.item_latent_q.log_prob(item_latent_sample)  # (num_seeds, num_items)
        return log_q_user.sum(dim=-1) + log_q_item.sum(dim=-1)

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
        # 1. sample latent from variational distribution Q.
        # (num_seeds, num_{users, items}, emb_dim)
        user_latent_sample_q = self.user_latent_q.reparameterize_sample(num_seeds)
        item_latent_sample_q = self.item_latent_q.reparameterize_sample(num_seeds)

        # 2. compute log p(latent) prior.
        if self.obs2prior_item:
            item_features = batch.item_features
        else:
            item_features = None

        if self.obs2prior_user:
            user_features = batch.user_features
        else:
            user_features = None

        # (num_seeds,)
        log_prior = self.log_prior(user_latent_sample_q, item_latent_sample_q,
                                   user_features=user_features,
                                   item_features=item_features)
        elbo = log_prior.mean()  # average over Monte Carlo samples for expectation.

        # 3. compute the log likelihood log p(obs|latent).
        # (num_seeds, num_sessions, num_items)
        log_p_all_items = self.log_likelihood(batch, user_latent_sample_q, item_latent_sample_q)
        # (num_sessions, num_items), average over Monte Carlo samples for expectation at dim 0.
        log_p_all_items = log_p_all_items.mean(dim=0)

        # log_p_cond[*, session] = log prob of the item bought in this session.
        # (num_sessions,)
        log_p_chosen_items = log_p_all_items[torch.arange(len(batch)), batch.label]
        elbo += log_p_chosen_items.sum(dim=-1)  # sessions are independent.

        # 4. optionally add log likelihood under variational distributions q(latent).
        if self.trace_log_q:
            log_q = self.log_variational(user_latent_sample_q, item_latent_sample_q)  # (num_seeds,)
            elbo -= log_q.mean()

        # log performance metrics.
        self.running_performance_dict['log_likelihood'].append(log_p_chosen_items.mean().detach().cpu().item())
        self.running_performance_dict['accuracy'].append(self.get_within_category_accuracy(log_p_all_items, batch.label))

        return elbo

"""Draft for the BEMB model"""
import os
from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import log_softmax
from torch_scatter import scatter_max
from torch_scatter.composite import scatter_log_softmax

from gaussian import batch_factorized_gaussian_log_prob

# TODO(Tianyu): change attribute names like log_p --> log_prob, likelihood etc.


class FactorizedGaussian(nn.Module):
    def __init__(self, num_classes: int, dim_out: int):
        # num_classes: number of items/users.
        # dim_out: embedding dimension.
        super(FactorizedGaussian, self).__init__()
        self.num_classes = num_classes
        self.dim_out = dim_out
        self.mean = nn.Parameter(torch.zeros(num_classes, dim_out), requires_grad=True)
        self.logstd = nn.Parameter(torch.ones(num_classes, dim_out), requires_grad=True)

    @property
    def device(self):
        return self.mean.device

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # input shape (batch_size, num_classes, dim_out)
        # output shape (batch_size, num_classes)
        return batch_factorized_gaussian_log_prob(self.mean, self.logstd, value)

    def reparameterize_sample(self, num_seeds: int=1) -> torch.Tensor:
        """Samples from the multivariate Gaussian distribution using the reparameterization trick.

        Args:
            num_seeds (int): number of samples generated.

        Returns:
            torch.Tensor: [description]
        """
        # TODO(Tianyu): think about this, can we use the same epsilon?
        #  using different epsilons should be more robust.
        eps = torch.randn(num_seeds, self.num_classes, self.dim_out).to(self.device)
        # parameters for each Gaussian distribution.
        mu = self.mean.view(1, self.num_classes, self.dim_out)
        std = torch.exp(self.logstd).view(1, self.num_classes, self.dim_out)
        out = mu + std * eps
        assert out.shape == (num_seeds, self.num_classes, self.dim_out)
        return out


class LearnableGaussian(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, std: Union[str, float]=0.1):
        """Construct a Gaussian distribution for prior of embeddings.
        p(alpha_ik | H_k, obsItem_i) = Gaussian( mean=H_k*obsItem_i, variance=s2obsPrior )
        p(beta_ik | H'_k, obsItem_i) = Gaussian( mean=H'_k*obsItem_i, variance=s2obsPrior )
        Args:
            dim_in (int): [description]
            dim_out (int): [description]
            std (Union[str, float]):
        """
        super(LearnableGaussian, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.H = nn.Linear(dim_in, dim_out, bias=False)
        if std == 'learnable_scalar':
            self.logstd = nn.Parameter(torch.randn(1), requires_grad=True)
        elif isinstance(std, float):
            self.logstd = torch.log(torch.scalar_tensor(std))
        else:
            raise NotImplementedError

    def log_prob(self, x_obs: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # x_obs: (num_classes, dim_in)
        # value: (batch_size, num_classes, dim_out)
        mu = self.H(x_obs)  # (num_classes, self.dim_out)
        # TODO(Tianyu): check here.
        logstd = self.logstd.expand(x_obs.shape[0], self.dim_out).to(x_obs.device)  # (num_classes, self.dim_out)
        # output shape (batch_size, num_classes)
        return batch_factorized_gaussian_log_prob(mu, logstd, value)


class BEMB(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 latent_dim: int,
                 trace_log_q: bool=False,
                 category_to_item: Dict[str, List[int]]=None,
                 likelihood_method: str='within_category',
                 learnable_user_prior: bool=False,
                 num_user_features: Optional[int]=None,
                 learnable_item_prior: bool=False,
                 num_item_features: Optional[int]=None
                 ):
        super(BEMB, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.trace_log_q = trace_log_q
        # maps from category IDs to IDs of items belonging to this category.
        self.likelihood_method = likelihood_method


        assert self.likelihood_method in ['all', 'within_category']
        if self.likelihood_method == 'within_category':
            self.category_to_item = category_to_item
            self.num_categories = len(self.category_to_item)

            category_idx = torch.zeros(self.num_items)
            for c, items_in_c in self.category_to_item.items():
                category_idx[items_in_c] = c
            self.category_idx = category_idx.long()

        # Q: build learnable variational distributions.
        self.user_latent_q = FactorizedGaussian(num_users, latent_dim)
        self.item_latent_q = FactorizedGaussian(num_items, latent_dim)

        # P: construct learnable priors for latents, if observables enter prior of latent.
        self.learnable_user_prior = learnable_user_prior
        if self.learnable_user_prior:
            self.user_latent_prior = LearnableGaussian(dim_in=num_user_features, dim_out=latent_dim)

        self.learnable_item_prior = learnable_item_prior
        if self.learnable_item_prior:
            self.item_latent_prior = LearnableGaussian(dim_in=num_item_features, dim_out=latent_dim)

        # an internal tracker for for tracking performance across batches.
        self.running_performance_dict = {'accuracy': [],
                                         'log_likelihood': []}

    @property
    def num_params(self) -> int:
        return sum([p.numel() for p in self.parameters()])

    @property
    def device(self):
        return self.user_latent_q.device

    def _validate_args(self):
        raise NotImplementedError

    def load_params(self, path: str):
        """A helper function loads learned parameters from BEMB cpp to the current model.

        Args:
            path (str): the output path of BEMB cpp.
        """
        def load_cpp_tsv(file):
            df = pd.read_csv(os.path.join(path, file), sep='\t', index_col=0, header=None)
            return torch.Tensor(df.values[:, 1:])

        cpp_theta_mean = load_cpp_tsv('param_theta_mean.tsv')
        cpp_theta_std = load_cpp_tsv('param_theta_std.tsv')

        cpp_alpha_mean = load_cpp_tsv('param_alpha_mean.tsv')
        cpp_alpha_std = load_cpp_tsv('param_alpha_std.tsv')

        # theta user
        self.user_latent_q.mean = cpp_theta_mean.to(self.device)
        self.user_latent_q.logstd = torch.log(cpp_theta_std).to(self.device)
        # alpha item
        self.item_latent_q.mean = cpp_alpha_mean.to(self.device)
        self.item_latent_q.logstd = torch.log(cpp_alpha_std).to(self.device)

    def report_performance(self) -> dict:
        """Reports content in the internal performance tracker, this method empties trackers after reporting."""
        report = dict()
        for k, v in self.running_performance_dict.items():
            report[k] = np.mean(v)
            # reset tracker.
            self.running_performance_dict[k] = []
        return report

    def log_prior_latent(self, user_latent_sample, item_latent_sample, user_features=None, item_features=None):
        # input shape (num_seeds, num_{users, items}, emb_dim)
        # output shape (num_seeds,)
        if self.learnable_user_prior:
            # use the learnt prior distribution.
            log_prob_user = self.user_latent_prior.log_prob(user_features, user_latent_sample)
        else:
            # otherwise, use N(0, 1) prior.
            standard_gaussian = MultivariateNormal(loc=torch.zeros(self.latent_dim).to(self.device),
                                                   covariance_matrix=torch.eye(self.latent_dim).to(self.device))
            log_prob_user = standard_gaussian.log_prob(user_latent_sample)  # (num_seeds, num_users)

        if self.learnable_item_prior:
            # use the learnt prior distribution.
            log_prob_item = self.item_latent_prior.log_prob(item_features, item_latent_sample)
        else:
            # otherwise, use N(0, 1) prior.
            standard_gaussian = MultivariateNormal(loc=torch.zeros(self.latent_dim).to(self.device),
                                                   covariance_matrix=torch.eye(self.latent_dim).to(self.device))
            log_prob_item = standard_gaussian.log_prob(item_latent_sample)  # (num_seeds, num_items)

        # sum across different users and items.
        # output shape (num_seeds,)
        return log_prob_user.sum(dim=-1) + log_prob_item.sum(dim=-1)

    def log_q_latent(self, user_latent_sample, item_latent_sample):
        # log q(latent) from the variational family.
        # input shape: (num_seeds, num_*, dim)
        log_q_user = self.user_latent_q.log_prob(user_latent_sample)  # (num_seeds, num_users)
        log_q_item = self.item_latent_q.log_prob(item_latent_sample)  # (num_seeds, num_items)
        # output shape (num_seeds,)
        return log_q_user.sum(dim=-1) + log_q_item.sum(dim=-1)

    def forward(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def log_p_all_items(self, batch, user_latent_value, item_latent_value) -> torch.Tensor:
        assert hasattr(batch, 'user_onehot') and hasattr(batch, 'label')
        user_idx = torch.nonzero(batch.user_onehot, as_tuple=True)[1].to(self.device)
        # (num_seeds, num_users, num_items) )
        # TODO(Tianyu): add utility formation here. Add to forward?
        utility = torch.bmm(user_latent_value, torch.transpose(item_latent_value, 1, 2))
        # get the utility for choosing each items by the user belong to each session.
        utility_by_session = utility[:, user_idx, :]  # (num_seeds, num_sessions, num_items)

        # log p(choosing item i | user, item latents)
        # TODO(Tianyu): change here.
        if self.likelihood_method == 'all':
            log_p = log_softmax(utility_by_session, dim=-1)
        elif self.likelihood_method == 'within_category':
            # compute log softmax separately within each category.
            log_p = scatter_log_softmax(utility_by_session, self.category_idx.to(self.device), dim=-1)
        # output shape: (num_seeds, num_sessions, self.num_items)
        return log_p

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

    def elbo(self, batch, num_seeds: int=1):
        # take expectation over q(latent), sample from q using num_seeds Monte Carlo samples.
        # generate user_latents and item_latents for each seed
        # TODO(Tianyu): should we use the same sample seeds for U and I?
        # (num_seeds, num_{users, items}, emb_dim)
        user_latent_sample_q = self.user_latent_q.reparameterize_sample(num_seeds)
        item_latent_sample_q = self.item_latent_q.reparameterize_sample(num_seeds)

        # log p(latent) prior.
        # (num_seeds,)
        if self.learnable_item_prior:
            item_features = batch.item_features
        else:
            item_features = None

        if self.learnable_user_prior:
            user_features = batch.user_features
        else:
            user_features = None

        log_prior_latent = self.log_prior_latent(user_latent_sample_q, item_latent_sample_q,
                                                 user_features=user_features,
                                                 item_features=item_features)
        elbo = log_prior_latent.mean()  # average over Monte Carlo samples for expectation.

        # log p(obs|latent)
        # TODO(Tianyu): add user_latent_sample_q to batch.
        # (num_seeds, num_sessions, num_items)
        log_p_all_items = self.log_p_all_items(batch, user_latent_sample_q, item_latent_sample_q)
        # (num_sessions, num_items), average over Monte Carlo samples for expectation.
        log_p_all_items = log_p_all_items.mean(dim=0)
        # (num_sessions,)
        label = batch.label

        # log_p_cond[*, session] = log prob of the item bought in this session.
        # (num_sessions,)
        log_p_chosen_items = log_p_all_items[torch.arange(len(label)), batch.label]

        # debugging performance metrics.
        self.running_performance_dict['log_likelihood'].append(log_p_chosen_items.mean().detach().cpu().item())
        self.running_performance_dict['accuracy'].append(self.get_within_category_accuracy(log_p_all_items, label))

        elbo += log_p_chosen_items.sum(dim=-1)  # sessions are independent.

        # log q(latent) prior, optionally added.
        if self.trace_log_q:
            log_q = self.log_q_latent(user_latent_sample_q, item_latent_sample_q)  # (num_seeds,)
            elbo -= log_q.mean()

        return elbo

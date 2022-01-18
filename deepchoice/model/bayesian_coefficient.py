from typing import Optional, Union, Tuple

import torch
import torch.nn as nn

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal


class BayesianCoefficient(nn.Module):
    def __init__(self,
                 variation: str,
                 num_classes: int,
                 obs2prior: bool,
                 num_obs: int=0,
                 dim: int=1,
                 prior_variance: float=1.0
                 ):
        """
        The Bayesian coefficient object represents a learnable tensor mu_i in R^k, where i is from a family (e.g., user)
        so there are num_classes * num_obs learnables.
        """
        super(BayesianCoefficient, self).__init__()
        # do we use this at all? TODO: drop self.variation.
        assert variation in ['item', 'user', 'constant']

        self.variation = variation
        self.obs2prior = obs2prior
        if variation == 'constant':
            assert not obs2prior

        self.num_classes = num_classes
        self.num_obs = num_obs
        self.dim = dim  # the dimension of greek letter parameter.
        self.prior_variance = prior_variance
        assert self.prior_variance > 0

        # create prior distribution.
        if self.obs2prior:
            # the mean of prior distribution depends on observables.
            # old: self.prior_H = nn.Linear(num_obs, dim, bias=False)
            # initiate a Bayesian Coefficient with shape (dim, num_obs) standard Gaussian.
            self.prior_H = BayesianCoefficient(variation='constant', num_classes=dim, obs2prior=False,
                                               dim=num_obs, prior_variance=1.0)
        else:
            self.register_buffer('prior_zero_mean', torch.zeros(num_classes, dim))

        # self.prior_cov_factor = nn.Parameter(torch.zeros(num_classes, dim, 1), requires_grad=False)
        # self.prior_cov_diag = nn.Parameter(torch.ones(num_classes, dim), requires_grad=False)
        self.register_buffer('prior_cov_factor', torch.zeros(num_classes, dim, 1))
        self.register_buffer('prior_cov_diag', torch.ones(num_classes, dim) * self.prior_variance)

        # create variational distribution.
        self.variational_mean_flexible = nn.Parameter(torch.randn(num_classes, dim), requires_grad=True)
        self.variational_logstd = nn.Parameter(torch.randn(num_classes, dim), requires_grad=True)

        self.register_buffer('variational_cov_factor', torch.zeros(num_classes, dim, 1))

        # self.variational_distribution = LowRankMultivariateNormal(loc=self.variational_mean,
        #                                                           cov_factor=self.variational_cov_factor,
        #                                                           cov_diag=torch.exp(self.variational_logstd))

        self.variational_mean_fixed = None

        self._check_args()

    def _check_args(self):
        if self.obs2prior:
            assert self.num_obs > 0

    def __repr__(self) -> str:
        if self.obs2prior:
            prior_str = f'prior=N(H*X_obs(H shape={self.prior_H.prior_zero_mean.shape}, X_obs shape={self.prior_H.dim}), Ix{self.prior_variance})'
        else:
            prior_str = f'prior=N(0, I)'
        return f'BayesianCoefficient(num_classes={self.num_classes}, dimension={self.dim}, {prior_str})'

    def update_variational_mean_fixed(self, new_value: torch.Tensor):
        assert new_value.shape == self.variational_mean_flexible.shape
        del self.variational_mean_fixed
        self.register_buffer('variational_mean_fixed', new_value)

    @property
    def variational_mean(self):
        if self.variational_mean_fixed is None:
            return self.variational_mean_flexible
        else:
            return self.variational_mean_fixed + self.variational_mean_flexible

    def log_prior(self, sample: torch.Tensor,
                  H_sample: Optional[torch.Tensor]=None,
                  x_obs: Optional[torch.Tensor]=None):
        """

        Args:
            sample (torch.Tensor): Monte Carlo samples of the variable with shape (num_seeds, num_classes, dim).
            H_sample (Optional[torch.Tensor], optional): Monte Carlo samples of the weight in obs2prior term, with shape
                (num_seeds, dim, self.num_obs), only required if obs2prior == True. Defaults to None.
            x_obs (Optional[torch.Tensor], optional): observables for obs2prior with shape (num_classes, num_obs),
                only required if obs2prior == True. Defaults to None.

        Returns:
            [type]: [description]
        """
        # p(sample)
        num_seeds, num_classes, dim = sample.shape
        # shape (num_seeds, num_classes)
        if self.obs2prior:
            assert H_sample.shape == (num_seeds, dim, self.num_obs)
            assert x_obs.shape == (num_classes, self.num_obs)
            x_obs = x_obs.view(1, num_classes, self.num_obs).expand(num_seeds, -1, -1)
            H_sample = torch.transpose(H_sample, 1, 2)
            assert H_sample.shape == (num_seeds, self.num_obs, dim)
            mu = torch.bmm(x_obs, H_sample)
            assert mu.shape == (num_seeds, num_classes, dim)

        else:
            mu = self.prior_zero_mean
        # breakpoint()
        out = LowRankMultivariateNormal(loc=mu,
                                        cov_factor=self.prior_cov_factor,
                                        cov_diag=self.prior_cov_diag).log_prob(sample)
        # breakpoint()
        assert out.shape == (num_seeds, num_classes)
        return out

    def log_variational(self, sample=None):
        # self.variational_distribution = LowRankMultivariateNormal(loc=self.variational_mean,
        #                                                           cov_factor=self.variational_cov_factor,
        #                                                           cov_diag=torch.exp(self.variational_logstd))
        num_seeds, num_classes, dim = sample.shape
        out = self.variational_distribution.log_prob(sample)
        assert out.shape == (num_seeds, num_classes)
        return out

    def reparameterize_sample(self, num_seeds: int=1) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        # self.variational_distribution = LowRankMultivariateNormal(loc=self.variational_mean,
        #                                                           cov_factor=self.variational_cov_factor,
        #                                                           cov_diag=torch.exp(self.variational_logstd))
        # shape (num_seeds, self.num_classes, self.dim).
        value_sample = self.variational_distribution.rsample(torch.Size([num_seeds]))
        if self.obs2prior:
            # sample obs2prior H as well.
            H_sample = self.prior_H.reparameterize_sample(num_seeds=num_seeds)
            return (value_sample, H_sample)
        else:
            return value_sample

    @property
    def variational_distribution(self):
        return LowRankMultivariateNormal(loc=self.variational_mean,
                                         cov_factor=self.variational_cov_factor,
                                         cov_diag=torch.exp(self.variational_logstd))

    @property
    def device(self) -> torch.device:
        return self.variational_mean.device

"""
Bayesian Coefficient is the building block for the BEMB model.

Author: Tianyu Du
Update: Apr. 28, 2022
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal


class BayesianCoefficient(nn.Module):
    def __init__(self,
                 variation: str,
                 num_classes: int,
                 obs2prior: bool,
                 num_obs: Optional[int] = None,
                 dim: int = 1,
                 prior_mean: float = 0.0,
                 prior_variance: float = 1.0
                 ) -> None:
        """The Bayesian coefficient object represents a learnable tensor mu_i in R^k, where i is from a family (e.g., user, item)
            so there are num_classes * num_obs learnable weights in total.
            The prior distribution of mu_i is N(0, I) or N(H*X_obs(H shape=num_obs, X_obs shape=dim), Ix1).
            The posterior(i.e., variational) distribution of mu_i is a Gaussian distribution with learnable mean mu_i and unit covariance.
            The mean of the variational distribution consists of two parts:
                1. The fixed part, which is not learnable. This part is particularly useful when the researcher want to impose
                    some structure on the variational distribution. For example, the research might have some variational mean
                    learned from another model and wish to use BEMB to polish the learned mean.
                2. The flexible part, which is the main learnable part of the variational mean.

        Args:
            variation (str): the variation # TODO: this will be removed in the next version, after we have a complete
                test pipline.
            num_classes (int): number of classes in the coefficient. For example, if we have user-specific coefficients,
                `theta_user`, the `num_classes` should be the number of users. If we have item-specific coefficients,
                the the `num_classes` should be the number of items.
            obs2prior (bool): whether the mean of coefficient prior depends on the observable or not.
            num_obs (int, optional): the number of observables associated with each class. For example, if the coefficient
                if item-specific, and we have `obs2prior` set to True, the `num_obs` should be the number of observables
                for each item.
                Defaults to None.
            dim (int, optional): the dimension of the coefficient.
                Defaults to 1.
            prior_mean (float): the mean of the prior distribution of coefficient.
                Defaults to 0.0.
            prior_variance (float): the variance of the prior distribution of coefficient.
                Defaults to 1.0.
        """
        super(BayesianCoefficient, self).__init__()
        # do we use this at all? TODO: drop self.variation.
        assert variation in ['item', 'user', 'constant', 'category']

        self.variation = variation
        self.obs2prior = obs2prior
        if variation == 'constant' or variation == 'category':
            if obs2prior:
                raise NotImplementedError('obs2prior is not supported for constant and category variation at present.')

        self.num_classes = num_classes
        self.num_obs = num_obs
        self.dim = dim  # the dimension of greek letter parameter.
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance

        # assert self.prior_variance > 0

        # create prior distribution.
        if self.obs2prior:
            # the mean of prior distribution depends on observables.
            # initiate a Bayesian Coefficient with shape (dim, num_obs) standard Gaussian.
            self.prior_H = BayesianCoefficient(variation='constant', num_classes=dim, obs2prior=False,
                                               dim=num_obs, prior_variance=1.0)
        else:
            self.register_buffer(
                'prior_zero_mean', torch.zeros(num_classes, dim) + (self.prior_mean))

        # self.prior_cov_factor = nn.Parameter(torch.zeros(num_classes, dim, 1), requires_grad=False)
        # self.prior_cov_diag = nn.Parameter(torch.ones(num_classes, dim), requires_grad=False)
        self.register_buffer('prior_cov_factor',
                             torch.zeros(num_classes, dim, 1))
        self.register_buffer('prior_cov_diag', torch.ones(
            num_classes, dim) * self.prior_variance)

        # create variational distribution.
        self.variational_mean_flexible = nn.Parameter(
            torch.randn(num_classes, dim), requires_grad=True)
        self.variational_logstd = nn.Parameter(
            torch.randn(num_classes, dim), requires_grad=True)

        self.register_buffer('variational_cov_factor',
                             torch.zeros(num_classes, dim, 1))

        self.variational_mean_fixed = None

    def __repr__(self) -> str:
        """Constructs a string representation of the Bayesian coefficient object.

        Returns:
            str: the string representation of the Bayesian coefficient object.
        """
        if self.obs2prior:
            prior_str = f'prior=N(H*X_obs(H shape={self.prior_H.prior_zero_mean.shape}, X_obs shape={self.prior_H.dim}), Ix{self.prior_variance})'
        else:
            prior_str = f'prior=N(0, I)'
        return f'BayesianCoefficient(num_classes={self.num_classes}, dimension={self.dim}, {prior_str})'

    def update_variational_mean_fixed(self, new_value: torch.Tensor) -> None:
        """Updates the fixed part of the mean of the variational distribution.

        Args:
            new_value (torch.Tensor): the new value of the fixed part of the mean of the variational distribution.
        """
        assert new_value.shape == self.variational_mean_flexible.shape
        del self.variational_mean_fixed
        self.register_buffer('variational_mean_fixed', new_value)

    @property
    def variational_mean(self) -> torch.Tensor:
        """Returns the mean of the variational distribution.

        Returns:
            torch.Tensor: the current mean of the variational distribution with shape (num_classes, dim).
        """
        if self.variational_mean_fixed is None:
            return self.variational_mean_flexible
        else:
            return self.variational_mean_fixed + self.variational_mean_flexible

    def log_prior(self,
                  sample: torch.Tensor,
                  H_sample: Optional[torch.Tensor] = None,
                  x_obs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the logP_{Prior}(Coefficient Sample) for provided samples of the coefficient. The prior will either be a
        zero-mean Gaussian (if `obs2prior` is False) or a Gaussian with a learnable mean (if `obs2prior` is True).

        Args:
            sample (torch.Tensor): Monte Carlo samples of the variable with shape (num_seeds, num_classes, dim), where
                sample[i, :, :] corresponds to one sample of the coefficient.

            # arguments required only if `obs2prior == True`:
            H_sample (Optional[torch.Tensor], optional): Monte Carlo samples of the weight in obs2prior term, with shape
                (num_seeds, dim, self.num_obs), this is required if and only if obs2prior == True.
                Defaults to None.
            x_obs (Optional[torch.Tensor], optional): observables for obs2prior with shape (num_classes, num_obs),
                only required if and only if obs2prior == True.
                Defaults to None.

        Returns:
            torch.Tensor: the log prior of the variable with shape (num_seeds, num_classes).
        """
        # p(sample)
        num_seeds, num_classes, dim = sample.shape
        # shape (num_seeds, num_classes)
        if self.obs2prior:
            assert H_sample.shape == (num_seeds, dim, self.num_obs)
            assert x_obs.shape == (num_classes, self.num_obs)
            x_obs = x_obs.view(1, num_classes, self.num_obs).expand(
                num_seeds, -1, -1)
            H_sample = torch.transpose(H_sample, 1, 2)
            assert H_sample.shape == (num_seeds, self.num_obs, dim)
            mu = torch.bmm(x_obs, H_sample)
            assert mu.shape == (num_seeds, num_classes, dim)

        else:
            mu = self.prior_zero_mean
        out = LowRankMultivariateNormal(loc=mu,
                                        cov_factor=self.prior_cov_factor,
                                        cov_diag=self.prior_cov_diag).log_prob(sample)
        assert out.shape == (num_seeds, num_classes)
        return out

    def log_variational(self, sample: torch.Tensor) -> torch.Tensor:
        """Given a set of sampled values of coefficients, with shape (num_seeds, num_classes, dim), computes the
            the log probability of these sampled values of coefficients under the current variational distribution.

        Args:
            sample (torch.Tensor): a tensor of shape (num_seeds, num_classes, dim) containing sampled values of coefficients,
                where sample[i, :, :] corresponds to one sample of the coefficient.

        Returns:
            torch.Tensor: a tensor of shape (num_seeds, num_classes) containing the log probability of provided samples
                under the variational distribution. The output is splitted by random seeds and classes, you can sum
                along the second axis (i.e., the num_classes axis) to get the total log probability.
        """
        num_seeds, num_classes, dim = sample.shape
        out = self.variational_distribution.log_prob(sample)
        assert out.shape == (num_seeds, num_classes)
        return out

    def rsample(self, num_seeds: int = 1) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Samples values of the coefficient from the variational distribution using re-parameterization trick.

        Args:
            num_seeds (int, optional): number of values to be sampled. Defaults to 1.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: if `obs2prior` is disabled, returns a tensor of shape (num_seeds, num_classes, dim)
                where each output[i, :, :] corresponds to one sample of the coefficient.
                If `obs2prior` is enabled, returns a tuple of samples: (1) a tensor of shape (num_seeds, num_classes, dim) containing
                sampled values of coefficient, and (2) a tensor o shape (num_seeds, dim, num_obs) containing samples of the H weight
                in the prior distribution.
        """
        value_sample = self.variational_distribution.rsample(
            torch.Size([num_seeds]))
        if self.obs2prior:
            # sample obs2prior H as well.
            H_sample = self.prior_H.rsample(num_seeds=num_seeds)
            return (value_sample, H_sample)
        else:
            return value_sample

    @property
    def variational_distribution(self) -> LowRankMultivariateNormal:
        """Constructs the current variational distribution of the coefficient from current variational mean and covariance.
        """
        return LowRankMultivariateNormal(loc=self.variational_mean,
                                         cov_factor=self.variational_cov_factor,
                                         cov_diag=torch.exp(self.variational_logstd))

    @property
    def device(self) -> torch.device:
        """Returns the device of tensors contained in this module."""
        return self.variational_mean.device

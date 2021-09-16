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


class BayesianCoefficient(nn.Module):
    def __init__(self,
                 variation: str,
                 num_classes: int,
                 obs2prior: bool,
                 num_obs: int=0,
                 dim: int=1,
                 ):
        super(BayesianCoefficient, self).__init__()
        assert variation in ['item', 'user', 'constant']
        if variation == 'constant':
            raise NotImplementedError

        self.variation = variation
        self.obs2prior = obs2prior
        self.num_classes = num_classes
        self.num_obs = num_obs
        self.dim = dim  # the dimension of greek letter parameter.

        # create prior distribution.
        if self.obs2prior:
            self.prior_distribution = LearnableGaussianPrior(dim_in=self.num_obs, dim_out=self.dim)
        else:
            # The dim_in doesn't matter for standard Gaussian.
            self.prior_distribution = StandardGaussianPrior(dim_in=self.num_obs, dim_out=self.dim)

        # create variational distribution.
        self.variational_distribution = VariationalFactorizedGaussian(num_classes=self.num_classes,
                                                                      dim=self.dim)

        self._check_args()

    def _check_args(self):
        if self.obs2prior:
            assert self.num_obs > 0

    def log_prior(self, sample: torch.Tensor, x_obs=None):
        # p(sample)
        num_seeds, num_classes, dim = sample.shape
        # shape (num_seeds, num_classes)
        out = self.prior_distribution.log_prob(x_obs, sample)
        assert out.shape == (num_seeds, num_classes)
        return out

    def log_variational(self, sample=None):
        num_seeds, num_classes, dim = sample.shape
        out = self.variational_distribution.log_prob(sample)
        assert out.shape == (num_seeds, num_classes)
        return out

    def reparameterize_sample(self, num_seeds: int=1):
        # shape (num_seeds, self.num_classes, self.dim).
        return self.variational_distribution.reparameterize_sample(num_seeds)

    # def __repr__(self):
    #     pass

    @property
    def device(self) -> torch.device:
        return self.variational_distribution.mean.device

    def __mul__(self, other):
        # is_monte_carlo =
        assert len(other.shape) == 5
        # shape = (num_seeds, num_sessions, num_users, num_items, num_params).

    def forward(self, other):
        # multipy self with obs entered.
        return ...

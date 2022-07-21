"""
Bayesian tensor object.
"""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class BayesianLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool=True,
                 W_variational_mean_fixed: Optional[torch.Tensor]=None,
                 device=None,
                 dtype=None,
                 W_prior_variance: float=1.0,
                 b_prior_variance: float=1.0
                 ):
        """Linear layer where weight and bias are modelled as distributions.
        """
        super().__init__()
        if dtype is not None:
            raise NotImplementedError('dtype is not Supported yet.')

        self.in_features = in_features  # the same as number of classes before.
        self.out_features = out_features  # the same as latent dimension before.
        self.bias = bias

        # ==============================================================================================================
        # prior distributions for mean and bias.
        # ==============================================================================================================
        # the prior of weights are gausssian distributions independent across in_feature dimensions.
        self.register_buffer('W_prior_mean', torch.zeros(in_features, out_features))
        self.register_buffer('W_prior_logstd', torch.ones(in_features, out_features) * np.log(W_prior_variance))

        if self.bias:
            self.register_buffer('b_prior_mean', torch.zeros(in_features, out_features))
            self.register_buffer('b_prior_logstd', torch.ones(in_features, out_features) * np.log(b_prior_variance))

        # ==============================================================================================================
        # variational distributions for weight and bias.
        # ==============================================================================================================
        if W_variational_mean_fixed is None:
            self.W_variational_mean_fixed = None
        else:
            assert W_variational_mean_fixed.shape == (in_features, out_features), \
                f'W_variational_mean_fixed tensor should have shape (in_features, out_features), got {W_variational_mean_fixed.shape}'
            self.register_buffer('W_variational_mean_fixed', W_variational_mean_fixed)

        # TODO: optionally add customizable initialization here.
        self.W_variational_mean_flexible = nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)
        self.W_variational_logstd = nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)

        if self.bias:
            self.b_variational_mean = nn.Parameter(torch.randn(out_features), requires_grad=True)
            self.b_variational_logstd = nn.Parameter(torch.randn(out_features), requires_grad=True)

        if device is not None:
            self.to(device)

        self.W_sample = None
        self.b_sample = None
        self.num_seeds = None

    @property
    def W_variational_mean(self):
        if self.W_variational_mean_fixed is None:
            return self.W_variational_mean_flexible
        else:
            return self.W_variational_mean_fixed + self.W_variational_mean_flexible

    def rsample(self, num_seeds: int=1) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """sample all parameters using re-parameterization trick.
        """
        self.num_seeds = num_seeds
        self.W_sample = self.W_variational_distribution.rsample(torch.Size([num_seeds]))

        if self.bias:
            self.b_sample = self.b_variational_distribution.rsample(torch.Size([num_seeds]))

        return self.W_sample, self.b_sample

    def dsample(self):
        """Deterministic sample method, set (W, b) sample to the mean of variational distribution."""
        self.num_seeds = 1
        self.W_sample = self.W_variational_mean.unsqueeze(dim=0)
        if self.bias:
            self.b_sample = self.b_variational_mean.unsqueeze(dim=0)
        return self.W_sample, self.b_sample

    def forward(self, x, mode: str='multiply'):
        """
        Forward with weight sampling. Forward does out = XW + b, for forward() method behaves like the embedding layer
        in PyTorch, use the lookup() method.
        To have determinstic results, call self.dsample() before executing.
        To have stochastic results, call self.rsample() before executing.
        mode in ['multiply', 'lookup']

        output shape: (num_seeds, batch_size, out_features).
        """
        assert self.num_seeds is not None, 'run BayesianLinear.rsample() or dsample() first to sample weight and bias.'

        # if determinstic, num_seeds is set to 1.
        # w: (num_seeds, in_features=num_classes, out_features)
        # b: (num_seeds, out_features)
        # x: (N, in_features) if multiply and (N,) if lookup.
        # output: (num_seeds, N, out_features)

        if mode == 'multiply':
            x = x.view(1, -1, self.in_features).expand(self.num_seeds, -1, -1)  # (num_seeds, N, in_features)
            out = x.bmm(self.W_sample)  # (num_seeds, N, out_features)
        elif mode == 'lookup':
            out = self.W_sample[:, x, :]  # (num_seeds, N, out_features)
        else:
            raise ValueError(f'mode={mode} is not allowed.')

        if self.bias:
            out += self.b_sample.view(self.num_seeds, 1, self.out_features)

        # (num_seeds, N, out_features)
        return out

    @property
    def W_variational_distribution(self):
        """the weight variational distribution."""
        return Normal(loc=self.W_variational_mean, scale=torch.exp(self.W_variational_logstd))

    @property
    def b_variational_distribution(self):
        return Normal(loc=self.b_variational_mean, scale=torch.exp(self.b_variational_logstd))

    @property
    def device(self) -> torch.device:
        return self.W_variational_mean.device

    def log_prior(self):
        """Evaluate the likelihood of the provided samples of parameter under the current prior distribution."""
        assert self.num_seeds is not None, 'run BayesianLinear.rsample() or dsample() first to sample weight and bias.'
        num_seeds = self.W_sample.shape[0]
        total_log_prob = torch.zeros(num_seeds, device=self.device)
        # log P(W_sample). shape = (num_seeds,)
        W_prior = Normal(loc=self.W_prior_mean, scale=torch.exp(self.W_prior_logstd))
        total_log_prob += W_prior.log_prob(self.W_sample).sum(dim=[1, 2])

        # log P(b_sample) if applicable.
        if self.bias:
            b_prior = Normal(loc=self.b_prior_mean, scale=torch.exp(self.b_prior_logstd))
            total_log_prob += b_prior.log_prob(self.b_sample).sum(dim=1)

        assert total_log_prob.shape == (num_seeds,)
        return total_log_prob

    def log_variational(self):
        """Evaluate the likelihood of the provided samples of parameter under the current variational distribution."""
        assert self.num_seeds is not None, 'run BayesianLinear.rsample() or dsample() first to sample weight and bias.'
        num_seeds = self.W_sample.shape[0]

        total_log_prob = torch.zeros(num_seeds, device=self.device)
        total_log_prob += self.W_variational_distribution.log_prob(self.W_sample).sum(dim=[1, 2])
        if self.bias:
            total_log_prob += self.b_variational_distribution.log_prob(self.b_sample).sum(dim=1)
        assert total_log_prob.shape == (num_seeds,)
        return total_log_prob

    def __repr__(self):
        prior_info = f'W_prior ~ N(mu={self.W_prior_mean}, logstd={self.W_prior_logstd})'
        if self.bias:
            prior_info += f'b_prior ~ N(mu={self.b_prior_mean}, logstd={self.b_prior_logstd})'
        return f"BayesianLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias}, {prior_info})"

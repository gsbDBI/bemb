"""
Bayesian tensor object.
Objective: this is a generalization of the Bayesian Coefficient object.
The Bayesian Tensor is designed to be hierarchical, so it's more than a single tensor, it's a module.
TODO: might change to Bayesian Layer or other name.

For the current iteration, we assume each entry of the weight matrix follows independent normal distributions.
TODO: might generalize this setting in the future.
TODO: generalize this setting to arbitrary shape tensors.
"""
from typing import Optional

import torch
import torch.nn as nn

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal


class BayesianLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool=True,
                 obs2prior: bool=False,
                 num_obs: Optional[int]=None,
                 device=None,
                 dtype=None):
        """Linear layer where weight and bias are modelled as distributions.
        """
        if dtype is not None:
            raise NotImplementedError('dtype is not Supported yet.')

        self.in_features = in_features  # the same as number of classes before.
        self.out_features = out_features  # the same as latent dimension before.
        self.bias = bias
        self.obs2prior = obs2prior
        self.num_obs = num_obs  # used only if obs2prior is True.

        # ==============================================================================================================
        # prior distributions for mean and bias.
        # ==============================================================================================================
        if self.obs2prior:
            self.prior_H = BayesianLinear(self.num_obs, self.out_features, bias=False, obs2prior=False)
            # TODO: optionally we make the std of prior learnable as well.
            if self.bias:
                raise NotImplementedError()
        else:
            # the prior of weights are gausssian distributions independent across in_feature dimensions.
            self.register_buffer('prior_weight_mean', torch.zeros(in_features, out_features))
            if self.bias:
                raise NotImplementedError()

        PRIOR_STD = 1.0  # TODO: add this to configuration.
        self.register_buffer('prior_weight_logstd', torch.ones(in_features, out_features) * torch.log(PRIOR_STD))

        # ==============================================================================================================
        # variational distributions for weight and bias.
        # ==============================================================================================================
        # TODO: optionally add initialization here.
        self.variational_weight_mean = nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)
        self.variational_weight_logstd = nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)

        if self.bias:
            raise NotImplementedError()
            self.variational_bias_mean = nn.Parameter(torch.randn(out_features), requires_grad=True)
            self.variational_bias_logstd = nn.Parameter(torch.randn(out_features), requires_grad=True)

        if device is not None:
            self.to(device)


    def _rsample_parameters(self, num_seeds: int=1):
        """sample weight and bias for forward() or lookup() method."""
        # sample weight
        w = self.weight_distribution.rsample(torch.Size([num_seeds]))
        if self.bias:
            b = self.bias_distribution.rsample(torch.Size([num_seeds]))
        else:
            b = None

        return w, b

    def rsample(self, num_seeds: int=1):
        """sample all parameters using reparameterization trick."""
        W = self.weight_distribution.rsample(torch.Size([num_seeds]))

        if self.bias:
            raise NotImplementedError()
            b = self.bias_distribution.rsample(torch.Size([num_seeds]))
        else:
            b = None

        if self.obs2prior:
            H, _, _ = self.prior_H.rsample(num_seeds)
        else:
            H = None

        # TODO: should we return a dictionary instead?
        return W, b, H

    def forward(self, x, num_seeds: int=1, deterministic: bool=False, mode: str='multiply'):
        """
        Forward with weight sampling. Forward does out = XW + b, for forward() method behaves like the embedding layer
        in PyTorch, use the lookup() method.
        If deterministic, use the mean.
        mode in ['multiply', 'lookup']
        """
        if deterministic:
            # set num_seeds to 1 so that we can reuse code from the non-deterministic version.
            num_seeds = 1
            raise NotImplementedError
        else:
            W, b, H = self.rsample(num_seeds)

        # if determinstic, num_seeds is set to 1.
        # w: (num_seeds, in_features=num_classes, out_features)
        # b: (num_seeds, out_features)
        # x: (N, in_features) if multiply and (N,) if lookup.
        # output: (num_seeds, N, out_features)

        if mode == 'multiply':
            x = x.view(1, -1, self.in_features).expand(num_seeds, -1, -1)  # (num_seeds, N, in_features)
            out = x.bmm(W)  # (num_seeds, N, out_features)
        elif mode == 'lookup':
            out = W[:, x, :]  # (num_seeds, N, out_features)
        else:
            raise Exception

        if self.bias:
            raise NotImplementedError()
            out += b.view(num_seeds, 1, self.out_features)

        if deterministic:
            # (N, out_features)
            return out.view(x.shape[0], self.out_features)
        else:
            # (num_seeds, N, out_features)
            return out

    @property
    def weight_distribution(self):
        """the weight variational distribution."""
        return Normal(loc=self.variational_weight_mean,
                      scale=torch.exp(self.variational_weight_logstd))

    @property
    def bias_distribution(self):
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        return self.variational_weight_mean.device

    def log_prior(self,
                  W_sample: torch.Tensor,
                  b_sample: Optional[torch.Tensor]=None,
                  H_sample: Optional[torch.Tensor]=None,
                  x_obs: Optional[torch.Tensor]=None):
        """Evaluate the likelihood of the provided samples of parameter under the current prior distribution."""
        if self.bias:
            raise NotImplementedError()

        num_seeds = W_sample.shape[0]
        if self.obs2prior:
            assert x_obs.shape == (self.in_features, self.num_obs)
            # TODO: change deterministic to False and do the sampling here as well.
            mu = self.prior_H(x_obs, determinstic=True, mode='multiply')  # (in_features, out_features)
        else:
            mu = self.prior_weight_mean

        total_log_prob = torch.zeros(num_seeds, device=self.device)
        # log P(W_sample). shape = (num_seeds)
        W_log_prob = Normal(loc=mu, scale=torch.exp(self.prior_weight_std)).log_prob(W_sample).sum(dim=[1, 2])
        total_log_prob += W_log_prob

        # log P(b_sample) if applicable.
        if self.bias:
            raise NotImplementedError()

        # log P(H_sample) if applicable.
        if self.obs2prior:
            H_log_prob = self.prior_H.log_prior(W_sample=H_sample)
            total_log_prob += H_log_prob

        assert total_log_prob.shape == (num_seeds)
        return total_log_prob

    def log_variational(self,
                        W_sample: torch.Tensor,
                        b_sample: Optional[torch.Tensor]=None):
        """Evaluate the likelihood of the provided samples of parameter under the current variational distribution."""
        num_seeds = W_sample.shape[0]
        W_log_prob = self.weight_distribution.log_prob(W_sample).sum(dim=[1, 2])
        if self.bias:
            raise NotImplementedError
        assert W_log_prob.shape == (num_seeds)

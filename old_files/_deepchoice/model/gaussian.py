"""Helper function to compute the log-likelihood of multivaraite Gaussian distribution."""
import math
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


def batch_factorized_gaussian_log_prob(mu, logstd, value):
    # constructs `num_classes` factorized Gaussian distribution from provided means and standard deviations.
    # mu: (num_classes, dim)
    # logstd: (num_classes, dim), only the diagonal of covariance matrix is needed.
    # values: (batch_size, num_classes, dim)
    # output: (batch_size, num_classes,)

    dim = mu.shape[-1]
    A = - 0.5 * dim * math.log(2 * math.pi)  # scalar.
    # - 1/2 * log det (cov).
    B = - 0.5 * logstd.sum(dim=-1).view(1, -1)  # (1, num_classes)
    num_classes = value.shape[1]
    # (x - mu)
    diff = value - mu.view(1, num_classes, dim)  # (batch_size, num_classes, dim)
    std = torch.exp(logstd).view(1, num_classes, dim)  # (1, num_classes, dim)
    # Sum[std * (x - mu)^2] across dimensions.
    C = - 0.5 * (1 / std * diff ** 2).sum(dim=-1)  # (batch_size, num_classes)
    return A + B + C


if __name__ == "__main__":
    # some tests
    batch_size = 32
    num_classes = 5
    dim = 10

    mu = torch.randn(num_classes, dim)
    logstd = torch.randn(num_classes, dim)
    values = torch.randn(batch_size, num_classes, dim)

    log_prob = torch.zeros(batch_size, num_classes)

    for i in range(num_classes):
        G = MultivariateNormal(loc=mu[i, :], covariance_matrix=torch.diag(torch.exp(logstd[i, :])))
        for j in range(batch_size):
            log_prob[j, i] = G.log_prob(values[j, i, :])

    log_prob_batch = batch_factorized_gaussian_log_prob(mu, logstd, values)

    print(log_prob == log_prob_batch)
    print(log_prob - log_prob_batch)
    print(f'max abs diff = {(log_prob - log_prob_batch).abs().max()}')

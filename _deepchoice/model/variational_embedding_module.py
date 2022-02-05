"""
A customized module template
"""

from pprint import pprint
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from _deepchoice.model.bayesian_coefficient import BayesianCoefficient
from termcolor import cprint
from torch_scatter import scatter_max
from torch_scatter.composite import scatter_log_softmax

class VariationalEmbeddingLayer(nn.Module):
    def __init__(self, num_classes: int, latent_dim: int):
        self.embedding_weight = BayesianCoefficient(variation='constant',
                                                    num_classes=num_classes,
                                                    obs2prior=False,
                                                    dim=latent_dim,
                                                    prior_variance=1.0)

    def forward(self, batch):
        return self.embedding_weight * batch.session_obs_w
        # batch.session_obs_w expected, (num_session, 1)
        # convert to batch.session_delta, (num_session, num_latent)
        batch.session_delta = self.emb(batch.session_obs_w)
        return batch

    # def __repr__(self):

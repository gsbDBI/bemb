from torch import nn
from bemb.model.bayesian_linear import BayesianLinear


class ExampleCustomizedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = BayesianLinear(in_features=7, out_features=1, bias=False)
        self.num_seeds = None

    def forward(self, batch):
        # return the utility level.
        day_of_week = batch.session_day_of_week[batch.session_index]
        utility = self.layer(day_of_week, mode='lookup')
        # assert utility.shape == (num_seeds, len(batch), 1)
        return utility.view(self.num_seeds, len(batch))

    def log_prior(self):
        return self.layer.log_prior()

    def log_variational(self):
        return self.layer.log_variational()

    def rsample(self, num_seeds: int):
        self.num_seeds = num_seeds
        return self.layer.rsample(num_seeds)

    def dsample(self):
        self.num_seeds = 1
        return self.layer.dsample()

import torch
import pytorch_lightning as pl
from _deepchoice.model import BEMB


class LitBEMB(pl.LightningModule):

    def __init__(self, learning_rate: float = 0.3, num_seeds: int=1, **kwargs):
        # use kwargs to pass parameter to BEMB Torch.
        super().__init__()
        self.model = BEMB(**kwargs)
        self.num_seeds = num_seeds
        self.learning_rate = learning_rate

    def __str__(self) -> str:
        return str(self.model)

    def training_step(self, batch, batch_idx):
        elbo = self.model.elbo(batch, num_seeds=self.num_seeds)
        self.log('train_elbo', elbo)
        loss = - elbo
        return loss

    def evaluation_step(self, batch, batch_idx, name):
        # common codes in the validation and test step.
        # name is either val or test.
        # get the log-likelihood.
        pred = self.model(batch)
        # non-differentiable metrics.
        performance = self.model.get_within_category_accuracy(pred, batch.label)
        performance[name + '_log_likelihood'] = pred[torch.arange(len(batch)), batch.label].mean().detach().cpu().item()
        for key, val in performance.items():
            self.log(name + '_' + key, val, prog_bar=(key == 'accuracy'))

    def validation_step(self, batch, batch_idx):
        self.evaluation_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluation_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

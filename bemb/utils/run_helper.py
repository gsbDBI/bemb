"""
This script contains a helper function for training and testing the BEMB model.
The helper function here serves as a template for the training procedure, we encourage users to make a copy of this
function and modify it to fully leverage the potential of pytorch lightning (e.g., early stopping and checkpointing).
"""
import time

import pytorch_lightning as pl
from torch_choice.data.utils import create_data_loader
from typing import List
from torch_choice.data import ChoiceDataset
from bemb.model import LitBEMBFlex


def section_print(input_text):
    """Helper function for printing"""
    print('=' * 20, input_text, '=' * 20)


def run(model: LitBEMBFlex, dataset_list: List[ChoiceDataset], batch_size: int=-1, num_epochs: int=10, num_workers: int=8, **kwargs) -> LitBEMBFlex:
    """A standard pipeline of model training and evaluation.

    Args:
        model (LitBEMBFlex): the initialized pytorch-lightning wrapper of bemb.
        dataset_list (List[ChoiceDataset]): train_dataset, validation_test, and test_dataset in a list of length 3.
        batch_size (int, optional): batch_size for training and evaluation. Defaults to -1, which indicates full-batch training.
        num_epochs (int, optional): number of epochs for training. Defaults to 10.
        **kwargs: additional keyword argument for the pytorch-lightning Trainer.

    Returns:
        LitBEMBFlex: the trained bemb model.
    """
    # present a summary of the model received.
    section_print('model received')
    print(model)

    # present a summary of datasets received.
    section_print('data set received')
    print('[Training dataset]', dataset_list[0])
    print('[Validation dataset]', dataset_list[1])
    print('[Testing dataset]', dataset_list[2])

    # create pytorch dataloader objects.
    train = create_data_loader(dataset_list[0], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation = create_data_loader(dataset_list[1], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # WARNING: the test step takes extensive memory cost since it computes likelihood for all items.
    # we run the test step with a much smaller batch_size.
    test = create_data_loader(dataset_list[2], batch_size=batch_size // 10, shuffle=False, num_workers=num_workers)

    section_print('train the model')
    trainer = pl.Trainer(gpus=1 if ('cuda' in str(model.device)) else 0,  # use GPU if the model is currently on the GPU.
                         max_epochs=num_epochs,
                         check_val_every_n_epoch=1,
                         log_every_n_steps=1,
                         **kwargs)
    start_time = time.time()
    trainer.fit(model, train_dataloaders=train, val_dataloaders=validation)
    print(f'time taken: {time.time() - start_time}')

    section_print('test performance')
    trainer.test(model, dataloaders=test)
    return model

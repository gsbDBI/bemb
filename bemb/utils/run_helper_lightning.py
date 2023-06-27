"""
This is a template script for researchers to train the PyTorch-based model with minimal effort.
The researcher only needs to initialize the dataset and the model, this training template comes with default
hyper-parameters including batch size and learning rate. The researcher should experiment with different levels
of hyper-parameter if the default setting doesn't converge well.
"""
import time
from typing import Optional

import pytorch_lightning as pl
from torch_choice.data import ChoiceDataset
from torch_choice.data.utils import create_data_loader

from bemb.model import LitBEMBFlex


def section_print(input_text):
    """Helper function for printing"""
    print('=' * 20, input_text, '=' * 20)


def run(model: "LitBEMBFlex",
        dataset_train: ChoiceDataset,
        dataset_val: Optional[ChoiceDataset]=None,
        dataset_test: Optional[ChoiceDataset]=None,
        batch_size: int=-1,
        num_epochs: int=10,
        num_workers: int=0,
        device: Optional[str]=None,
        check_val_every_n_epoch: Optional[int]=None,
        **kwargs) -> "LitBEMBFlex":
    """_summary_

    Args:
        model (LitBEMBFlex): the initialized BEMB model.
        dataset_train (ChoiceDataset): the dataset for training.
        dataset_val (ChoiceDataset): an optional dataset for validation.
        dataset_test (ChoiceDataset): an optional dataset for testing.
        batch_size (int, optional): batch size for model training. Defaults to -1.
        num_epochs (int, optional): number of epochs for the training. Defaults to 10.
        num_workers (int, optional): number of parallel workers for data loading. Defaults to 0.
        device (Optional[str], optional): the device that trains the model, if None is specified, the function will
            use the current device of the provided model. Defaults to None.
        check_val_every_n_epoch (Optional[int], optional): the frequency of validation, if None is specified,
            validation will be performed every 10% of total epochs. Defaults to None.
        **kwargs: other keyword arguments for the pytorch lightning trainer, this is for users with experience in
            pytorch lightning and wish to customize the training process.

    Returns:
        LitBEMBFlex: the trained model with estimated parameters in it.
    """
    # ==================================================================================================================
    # Setup the lightning wrapper.
    # ==================================================================================================================
    lightning_model = model
    if device is None:
        # infer from the model device.
        device = str(model.device)
    # the cloned model will be used for standard error calculation later.
    # model_clone = deepcopy(model)
    section_print('model received')
    print(model)

    # ==================================================================================================================
    # Prepare the data.
    # ==================================================================================================================
    # present a summary of datasets received.
    section_print('data set received')
    print('[Train dataset]', dataset_train)
    print('[Validation dataset]', dataset_val)
    print('[Test dataset]', dataset_test)

    # create pytorch dataloader objects.
    train_dataloader = create_data_loader(dataset_train.to(device), batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if dataset_val is not None:
        val_dataloader = create_data_loader(dataset_val.to(device), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        val_dataloader = None

    if dataset_test is not None:
        test_dataloader = create_data_loader(dataset_test.to(device), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        test_dataloader = None

    # ==================================================================================================================
    # Training the model.
    # ==================================================================================================================
    # training BEMB is more complicated, don't add early stopping for now.
    callbacks = []

    trainer = pl.Trainer(accelerator="cuda" if "cuda" in device else device,  # note: "cuda:0" is not a accelerator name.
                         devices="auto",
                         max_epochs=num_epochs,
                         check_val_every_n_epoch=num_epochs // 10 if check_val_every_n_epoch is None else check_val_every_n_epoch,
                         log_every_n_steps=1,
                         callbacks=callbacks,
                         **kwargs)
    start_time = time.time()
    trainer.fit(lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print(f'Time taken for training: {time.time() - start_time}')
    if test_dataloader is not None:
        trainer.test(lightning_model, test_dataloader)
    else:
        print('Skip testing, no test dataset is provided.')

    return model

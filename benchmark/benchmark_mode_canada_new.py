"""
Benchmark using the ModeCanada dataset in R.
"""
import argparse
import os
import pdb
import sys
from typing import Optional
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
import yaml

sys.path.append('./src')
sys.path.append('../src')
sys.path.append('./')
sys.path.append('../src/model')
from data.data_formatter import stata_to_tensors
# from data.dataset import CMDataset
from data.choice_dataset import ChoiceDataset
from model.conditional_logit_model_v2 import ConditionalLogitModel
from train.train import train


def main(modifier: Optional[dict]=None):
    # arg_path = sys.argv[1]
    arg_path = './args_mode_canada_new.yaml'
    with open(arg_path) as file:
        args = argparse.Namespace(**yaml.load(file, Loader=yaml.FullLoader))  # unwrap dictionary loaded.

    # if modifier is not None:
    #     for k, v in modifier.items():
    #         setattr(args, k, v)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if args.device != 'cpu':
        print(f'Running on device: {torch.cuda.get_device_name(args.device)}')
    else:
        print(f'Running on CPU.')

    project_path = '~/Development/deepchoice'
    mode_canada = pd.read_csv(os.path.join(project_path, 'data/ModeCanada.csv'), index_col=0)
    mode_canada = mode_canada.query('noalt == 4').reset_index(drop=True)

    num_sessions = mode_canada['case'].nunique()
    print(f'{num_sessions=:}')
    
    from typing import Union, List

    def pivot3d(df: pd.DataFrame, dim0: str, dim1: str, values: Union[str, List[str]]) -> torch.Tensor:
        """
        Creates a tensor of shape (df[dim0].nunique(), df[dim1].nunique(), len(values)) from the
        provided data frame.

        Example, if dim0 is the column of session ID, dim1 is the column of alternative names, then
            out[t, i, k] is the feature values[k] of item i in session t. The returned tensor
            has shape (num_sessions, num_items, num_params), which fits the purpose of conditioanl
            logit models.
        """
        if not isinstance(values, list):
            values = [values]
        
        dim1_list = sorted(df[dim1].unique())
        
        tensor_slice = list()
        for value in values:
            layer = df.pivot(index=dim0, columns=dim1, values=value)
            tensor_slice.append(torch.Tensor(layer[dim1_list].values))
        
        tensor = torch.stack(tensor_slice, dim=-1)
        assert tensor.shape == (df[dim0].nunique(), df[dim1].nunique(), len(values))
        return tensor

    price_cost = pivot3d(mode_canada, 'case', 'alt', ['cost', 'freq', 'ovt'])
    session_income = torch.Tensor(mode_canada.groupby('case')['income'].first().values).view(-1, 1)
    price_ivt = pivot3d(mode_canada, 'case', 'alt', 'ivt')
    
    label = mode_canada.pivot('case', 'alt', 'choice')[['air', 'bus', 'car', 'train']].values
    _, label = torch.nonzero(torch.Tensor(label), as_tuple=True)
    label = label.long()
    
    user_onehot = torch.ones(num_sessions, 1).long()

    # dataset = CMDataset(X=X, user_onehot=user_onehot, A=A, Y=Y, C=C, device=args.device)
    dataset = ChoiceDataset(label=label, user_onehot=user_onehot,
                            price_cost=price_cost,
                            price_ivt=price_ivt,
                            session_income=session_income).to(args.device)
 
    print('Building the model...')
    model = ConditionalLogitModel(num_items=4,
                                  num_users=1,
                                  coef_variation_dict=args.coef_variation_dict,
                                  num_param_dict=args.num_param_dict)

    model = model.to(args.device)
    print(f'Number of parameters: {model.num_params}')
    
    if args.batch_size == -1:
        # use full-batch.
        args.batch_size = len(dataset)

    sampler = BatchSampler(
        RandomSampler(dataset) if args.shuffle else SequentialSampler(dataset),
        batch_size=args.batch_size,
        drop_last=False)
    # feed a batch_sampler as sampler so that dataset.__getitem__ is called with a list of indices.
    # cannot use multiple workers if the entire dataset is already on GPU.
    data_train = torch.utils.data.DataLoader(dataset,
                                             sampler=sampler,
                                             num_workers=0,  # 0 if dataset.device == 'cuda' else os.cpu_count(),
                                             collate_fn=lambda x: x[0],
                                             pin_memory=(dataset.device == 'cpu'))
    start = datetime.now()
    train([data_train, None, None], model, args)

    # print final estimation.
    print('intercept')
    print(model.coef_dict['intercept'].coef)
    print('cost, freq, ovt')
    print(model.coef_dict['price_cost'].coef)
    print('income')
    print(model.coef_dict['session_income'].coef)
    print('ivt')
    print(model.coef_dict['price_ivt'].coef)

    print('=' * 10 + 'computing Hessian' + '=' * 10)
    std_dict = model.compute_std(x_dict=dataset.x_dict,
                                 availability=dataset.item_availability,
                                 user_onehot=dataset.user_onehot,
                                 y=dataset.label)

    for var_type in std_dict.keys():
        print(f'Variable Type: {var_type}')
        for i, s in enumerate(std_dict[var_type]):
            c = model.coef_dict[var_type].coef.view(-1,)[i]
            print(f'{c.item():.4f}, {s.item():.4f}')

    print(f'Total time taken: {datetime.now() - start}')


if __name__ == '__main__':
    main()

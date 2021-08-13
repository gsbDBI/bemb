"""
A general procedure of using the deepchoice library.
"""
import argparse
import sys
from datetime import datetime

import deepchoice
import yaml
from deepchoice.data.contrib.mode_canada import main as load_dataset
from deepchoice.data.utils import create_data_loader
from deepchoice.model import ConditionalLogitModel
from deepchoice.train.train import train

if __name__ == '__main__':
    # 0. load configuration.
    with open(sys.argv[1], 'r') as file:
        args = argparse.Namespace(**yaml.load(file, Loader=yaml.FullLoader))  # unwrap dictionary loaded.

    # 1. load dataset.
    dataset = load_dataset(args)
 
    # 2. split dataset.
    # Not Implemented Yet.
    datasets = [dataset, None, None]

    # 3. create dataloader.
    data_loaders = [create_data_loader(dataset, args), None, None]

    # 4. create model.
    model = ConditionalLogitModel(num_items=4,
                                  num_users=1,
                                  coef_variation_dict=args.coef_variation_dict,
                                  num_param_dict=args.num_param_dict)

    model = model.to(args.device)

    # 5. training loop.
    start = datetime.now()
    train(data_loaders, model, args)

    # 6. post-training analysis.
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

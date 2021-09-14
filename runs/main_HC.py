import argparse
import sys
from datetime import datetime

import deepchoice
import yaml
from deepchoice.data.contrib.HC import main as load_dataset
from deepchoice.data.utils import create_data_loader
from deepchoice.model import NestedLogitModel
from deepchoice.train.train import train


args = argparse.Namespace(data_path='~/Development/deepchoice/data')

dataset = load_dataset(args)

item_names = ['ec', 'ecc', 'er', 'erc', 'gc', 'gcc', 'hpc']
category_to_item = {'with_cooling': [0, 2, 4], 'no_cooling': [1, 3, 5, 6]}
for k, v in category_to_item.items():
    print(f'Category={k}:')
    print([f'[{i}] {item_names[i]}' for i in v])


model = NestedLogitModel(category_to_item=category_to_item,
                         category_coef_variation_dict={},
                         category_num_param_dict={},
                         item_coef_variation_dict={'item_obs': 'item'},
                         item_num_param_dict={'item_obs': 7})

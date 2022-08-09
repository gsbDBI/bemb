"""
This scripts contain unit tests validating functionalities of data containers.

Author: Tianyu Du
Date: Aug. 6, 2022
"""
import unittest
import itertools

import numpy as np
import pandas as pd
import torch
from bemb.model import BEMBFlex
import simulate_choice_dataset

global numUs, num_items, data_size
num_users = 50
num_items = 100
data_size = 10000
num_seeds = 32


class TestBEMBFlex(unittest.TestCase):
    """
    Testing core functionality of bemb.
    """
    # def __init__(self):
    #     pass

    # def test_initialization(self):
    #    pass

    # def test_estimation(self):
    #     pass

    # ==================================================================================================================
    # Test Arguments and Options in the Initialization Method
    # ==================================================================================================================
    def test_init(self):
        pass

    def test_H_zero_mask(self):
        pass

    # ==================================================================================================================
    # Test API Methods
    # ==================================================================================================================
    def test_forward_shape(self):
        dataset_list = simulate_choice_dataset.simulate_dataset(num_users=num_users, num_items=num_items, data_size=data_size)
        batch = dataset_list[0]
        # test different variations of the forward function.
        # return_type X return_scope X deterministic X pred_items.
        for return_type, return_scope, deterministic, pred_item in itertools.product(['utility', 'log_prob'], ['item_index', 'all_items'], [True, False], [True, False]):
            if not pred_item:
                # generate fake binary labels.
                batch.label = torch.LongTensor(np.random.randint(2, size=len(batch)))

            # initialize the model.
            bemb = BEMBFlex(
                pred_item=pred_item,
                utility_formula='theta_user * alpha_item',
                num_users=num_users,
                num_items=num_items,
                num_classes=None if pred_item else 2,
                num_user_obs=batch.user_obs.shape[1],
                num_item_obs=batch.item_obs.shape[1],
                obs2prior_dict={'theta_user': True, 'alpha_item': True},
                coef_dim_dict={'theta_user': 10, 'alpha_item': 10}
            )

            output = bemb.forward(batch,
                                  return_type=return_type, return_scope=return_scope,
                                  deterministic=deterministic,
                                  sample_dict=None,
                                  num_seeds=num_seeds)

            if (return_scope == 'item_index') and (deterministic == True):
                self.assertEqual(output.shape, (len(batch),))
            elif (return_scope == 'all_items') and (deterministic == True):
                self.assertEqual(output.shape, (len(batch), num_items))
            elif (return_scope == 'item_index') and (deterministic == False):
                self.assertEqual(output.shape, (num_seeds, len(batch)))
            elif (return_scope == 'item_index') and (deterministic == False):
                self.assertEqual(output.shape, (num_seeds, len(batch), num_items))

    def test_predict_proba_shape(self):
        """
        Check shape of object returned by the predict_proba method.
        """
        dataset_list = simulate_choice_dataset.simulate_dataset(num_users=num_users, num_items=num_items, data_size=data_size)
        batch = dataset_list[-1]

        for pred_item in [True, False]:
            bemb = BEMBFlex(
                pred_item=pred_item,
                utility_formula='theta_user * alpha_item',
                num_users=num_users,
                num_items=num_items,
                num_classes=None if pred_item else 2,
                num_user_obs=dataset_list[0].user_obs.shape[1],
                num_item_obs=dataset_list[0].item_obs.shape[1],
                obs2prior_dict={'theta_user': True, 'alpha_item': True},
                coef_dim_dict={'theta_user': 10, 'alpha_item': 10}
            )
            P = bemb.predict_proba(batch)

            if pred_item:
                self.assertEqual(P.shape, (len(batch), num_items))
            else:
                self.assertEqual(P.shape, (len(batch), 2))

if __name__ == '__main__':
    unittest.main()

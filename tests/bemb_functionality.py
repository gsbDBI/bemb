"""
This scripts contain unit tests validating functionalities of data containers.

Author: Tianyu Du
Date: Aug. 6, 2022
"""
import unittest

import numpy as np
import pandas as pd
import torch
from bemb.model import BEMBFlex
import simulate_choice_dataset

global numUs, num_items, data_size
num_users = 50
num_items = 100
data_size = 10000


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

    def test_inference_predict_proba(self):
        """
        Check shape of object returned by the predict_proba method.
        """
        self.assertTrue(True)
        dataset_list = simulate_choice_dataset.simulate_dataset(num_users=num_users, num_items=num_items, data_size=data_size)

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
            batch = dataset_list[-1]
            P = bemb.predict_proba(batch)

            if pred_item:
                self.assertEqual(P.shape, (len(batch), num_items))
            else:
                self.assertEqual(P.shape, (len(batch), 2))
                self.assertTrue(torch.all(P.sum(dim=1) == 1))

if __name__ == '__main__':
    unittest.main()

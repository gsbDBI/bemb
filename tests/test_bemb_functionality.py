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
from bemb.model.bemb import parse_utility
import simulate_choice_dataset

global numUs, num_items, data_size
num_users = 50
num_items = 100
num_sessions = 500
data_size = 10000
num_seeds = 32


class TestUtilityParser(unittest.TestCase):
    def test_parser_and_model_creation(self):
        formula = 'intercept1_constant + intercept2_item + intercept3_user + alpha_item * beta_user + user_obs * gamma_user + item_obs * delta_item + comp_user * comp_item * user_obs'
        additive_decomposition = parse_utility(formula)
        self.assertTrue(additive_decomposition[0]['coefficient'] == ['intercept1_constant'])
        self.assertTrue(additive_decomposition[1]['coefficient'] == ['intercept2_item'])
        self.assertTrue(additive_decomposition[2]['coefficient'] == ['intercept3_user'])
        self.assertTrue(all(additive_decomposition[i]['observable'] is None for i in range(3)))

        self.assertTrue(additive_decomposition[3]['coefficient'] == ['alpha_item', 'beta_user'] and additive_decomposition[3]['observable'] is None)
        self.assertTrue(additive_decomposition[4]['coefficient'] == ['gamma_user'] and additive_decomposition[4]['observable'] == 'user_obs')
        self.assertTrue(additive_decomposition[5]['coefficient'] == ['delta_item'] and additive_decomposition[5]['observable'] == 'item_obs')
        self.assertTrue(additive_decomposition[6]['coefficient'] == ['comp_user', 'comp_item'] and additive_decomposition[6]['observable'] == 'user_obs')

    def test_parser_constant_and_null_coef(self):
        formula_1 = 'user_obs * gamma_constant + alpha_item * beta_item'
        formula_2 = 'user_obs * gamma + alpha_item * beta_item'
        u1 = parse_utility(formula_1)
        u2 = parse_utility(formula_2)
        self.assertTrue(u1 == u2)

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


class TestBEMBFlexV2(unittest.TestCase):
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
    def test_prediction_shapes(self):
        dataset_list = simulate_choice_dataset.simulate_dataset_v2(num_users=num_users, num_items=num_items, num_sessions=num_sessions, data_size=data_size)
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
                utility_formula="a_user + b_item + c_constant + d_user * e_item + f1_constant * user_obs + f2_constant * item_obs + f3_constant * session_obs + f4_constant * useritem_obs + f5_constant * usersession_obs + f6_constant * itemsession_obs + f7_constant * usersessionitem_obs",
                num_users=num_users,
                num_items=num_items,
                num_sessions=num_sessions,
                num_classes=None if pred_item else 2,
                num_user_obs=batch.user_obs.shape[1],
                num_item_obs=batch.item_obs.shape[1],
                obs2prior_dict={'a_user': True, 'b_item': True, 'c_constant': False, 'd_user': True, 'e_item': True,
                                'f1_constant': False, 'f2_constant': False, 'f3_constant': False, 'f4_constant': False, 'f5_constant': False, 'f6_constant': False, 'f7_constant': False},
                coef_dim_dict={'a_user': 1, 'b_item': 1, 'c_constant': 1, 'd_user': 10, 'e_item': 10,
                                'f1_constant': batch.user_obs.shape[-1], 'f2_constant': batch.item_obs.shape[-1], 'f3_constant': batch.session_obs.shape[-1],
                                'f4_constant': batch.useritem_obs.shape[-1] , 'f5_constant': batch.usersession_obs.shape[-1], 'f6_constant': batch.itemsession_obs.shape[-1], 'f7_constant': batch.usersessionitem_obs.shape[-1]}
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

            # test predict_proba method.
            P = bemb.predict_proba(batch)
            if pred_item:
                self.assertEqual(P.shape, (len(batch), num_items))
            else:
                self.assertEqual(P.shape, (len(batch), 2))
                self.assertTrue(torch.all(P.sum(dim=1) == 1))

if __name__ == '__main__':
    unittest.main()

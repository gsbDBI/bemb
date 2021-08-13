"""
The datalaoder for the logit model.
"""
import os
from typing import Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils import data


class LogitDataset(data.Dataset):
    """
    The original dataloader for the conditional logit model.
    """
    def __init__(self,
                 data_path: str,
                 category_we_care: Optional[Union[str, int]]=None,
                 split: str='train',
                 item_list=None):
        super().__init__()

        # load item prices at each time t, which includes (store, date).
        self.item_sess_price = pd.read_csv(os.path.join(data_path, 'item_sess_price.tsv'),
                                           sep='\t',
                                           dtype={'item_id': str, 'date': str, 'price': np.float64},
                                           names=['item_id', 'date', 'price'],
                                           header=None)

        # TODO(Tianyu): do we really need to make the copy.
        self.item_sess_price_copy = self.item_sess_price.copy()

        # item ID to category ID mapping.
        self.item_group = pd.read_csv(os.path.join(data_path, 'itemGroup.tsv'),
                                      sep='\t', header=None, dtype=str,
                                      names=['item_id', 'category_id'])

        # load user observable features.
        columns = ['user_id']
        # read column names from another file.
        columns.extend(pd.read_csv(
            f'{data_path}/obsUserNames.tsv', sep='\t', header=None)[0])
        dtypes = [str] + [np.int64] * (len(columns) - 1)
        dtypes = {a: b for a, b in zip(columns, dtypes)}
        self.obs_user = pd.read_csv(os.path.join(data_path, 'obsUser.tsv'),
                                    sep='\t', header=None,
                                    names=columns, dtype=dtypes)

        # load calendar day information to sessions.
        columns = ['date', 'wk', 'dayofweek', 'q']
        dtypes = [str] + [np.int64] * 3
        dtypes = {a: b for a, b in zip(columns, dtypes)}
        self.sess_days = pd.read_csv(os.path.join(data_path, 'sess_days.tsv'),
                                     sep='\t', header=None, names=columns, dtype=dtypes)

        # load the dataset of purchasing history.
        columns = ['user_id', 'item_id', 'date', 'num_purchase']
        dtypes = [str] * 3 + [np.int64]
        dtypes = {a: b for a, b in zip(columns, dtypes)}
        if (split != 'train'):
            assert item_list is not None,\
                'Please provide item list for test and validation.'
        split_file = split + '.tsv'  # get the file name.
        self.purchase = pd.read_csv(os.path.join(data_path, split_file),
                                    sep='\t', header=None, names=columns, dtype=dtypes)

        # load availability information.
        columns = ['date', 'item_id']
        self.availability_data = pd.read_csv(os.path.join(data_path, 'availabilityList.tsv'),
                                             sep='\t', header=None, names=columns, dtype=str)
        self.availability_data = self.availability_data.merge(self.item_group, on='item_id', how='left')
        
        # TODO(Tianyu): what's the use of availability_DATE? If not used, remove them.
        # self.availability_date = self.availability_data[['date']].drop_duplicates()
        # self.availability_date.columns = ['date']
        # self.availability_date = self.availability_date.sort_values(by=['date'])

        # add category into the main training set.
        self.joint_training_data = self.purchase.merge(self.item_group, on='item_id', how='left')

        self.category_we_care = category_we_care
        if self.category_we_care is not None:
            # filter to get the category of interest.
            self.curr_training_data = self.joint_training_data.loc[
                self.joint_training_data['category_id'] == str(self.category_we_care)]

            self.availability_data = self.availability_data.loc[
                self.availability_data['category_id'] == str(self.category_we_care)]
        else:
            # consider all categories.
            self.curr_training_data = self.joint_training_data
            self.category_we_care = self.curr_train_data['category_id'].unique().values

        # add calendar day information associated with session.
        self.curr_training_data = self.curr_training_data.merge(self.sess_days, on='date', how='left')

        # sorted in this order to ensure consistency among different tensors.
        self.curr_training_data.sort_values(['item_id', 'date'], inplace=True)

        self.training_item_ids = np.array(self.curr_training_data['item_id'])

        self.train_size = self.training_item_ids.shape[0]
        
        if item_list is not None:
            # only consider a specific collection of items.
            self.item_list = item_list
            self.availability_data = self.availability_data[
                self.availability_data['item_id'].isin(self.item_list)]  # One category each time
            # self.availability_data = self.availability_data.set_index(['date', 'item_id'])
            self.item_sess_price_copy = self.item_sess_price_copy[
                self.item_sess_price_copy['item_id'].isin(self.item_list)]  # One category each time
        else:
            self.item_list = sorted(set(self.training_item_ids))

        self.report_stats()

        # count purchasing history of each item. TODO(Tianyu): is this required?
        # _uniq = np.unique(self.training_item_ids, return_counts=True)
        # self.count_dict = dict(zip(_uniq[0], _uniq[1]))

        self.curr_training_data.sort_values(by=['date', 'user_id', 'item_id'], inplace=True)
        self.data_arr = np.array(self.curr_training_data)
        
        # construct session level features.
        (self.all_constants, self.all_obs_user, self.all_prices,
         self.all_item_onehot, self.all_month_indicator, self.all_day_indicator,
         self.all_week_indicator, self.all_availabilities) = self.get_all_data()
        _, _ = self.get_X_Y_all()  # test running.

    def report_stats(self):
        df = self.curr_training_data
        print('number of records: ', len(df))
        print('number of items: ', df['item_id'].nunique())
        print('number of users: ', df['user_id'].nunique())
        print('number of dates (calendar date X store ID): ', df['date'].nunique())
        print('number of session (user X date): ', len(df.groupby(['user_id', 'date'])))

    def __len__(self):
        return self.curr_training_data.shape[0]

    def get_all_data(self) -> Tuple[torch.Tensor]:
        """Returns a complete set of relevant data."""
        # get attributes and features in form of array.
        item_date = self.curr_training_data['date'].values
        item_id = self.curr_training_data['item_id'].values
        item_week = self.curr_training_data['wk'].values
        item_dayofweek = self.curr_training_data['dayofweek'].values
        # onehot encoding of item purchased at each session.
        item_onehot = np.array([
            np.array([1 if i == item_id_i else 0 for i in self.item_list])
            for item_id_i in item_id])

        # retrieve price information.
        self.item_sess_price_copy.sort_values(by=['date', 'item_id'], inplace=True)
        # item_dates = self.item_sess_price_copy[['date', 'item_id']]
        self.availability_data['availability'] = 1  # ['date', 'item_id', 'availability']
        
        a_wide = self.availability_data.pivot(index='date', columns='item_id', values='availability')
        a_wide.fillna(0, inplace=True)

        num_dates = self.purchase['date'].nunique()
        num_items = len(self.item_list)
        assert a_wide.shape == (num_dates, num_items)
        a_wide = a_wide.reset_index()  # now date is a column.
        A = self.curr_training_data.merge(a_wide, how='left', on='date')
        A.sort_values(by=['date', 'item_id'], inplace=True)
        A = A[self.item_list]
        assert A.shape == (len(self), num_items)

        self.item_sess_price_copy.reset_index(drop=True, inplace=True)
        prices_wide = self.item_sess_price_copy.pivot(index='date', columns='item_id', values='price')
        prices_wide.fillna(0, inplace=True)  # unknown prices.
        prices = self.curr_training_data.set_index('date').join(prices_wide, how='left').reset_index()
        prices.sort_values(by=['date', 'user_id'], inplace=True)
        prices = prices[list(self.item_list)]
        prices = np.array(prices)
        assert prices.shape == (len(self), len(self.item_list))

        obs_user = self.curr_training_data.set_index('user_id').join(self.obs_user.set_index('user_id'), how='left').reset_index()
        obs_user = self.curr_training_data.merge(self.obs_user, how='left', on='user_id')
        user_feature_columns = list(self.obs_user.columns)
        user_feature_columns.remove('user_id')
        obs_user = obs_user[user_feature_columns].values
        assert obs_user.shape == (len(self), len(user_feature_columns))

        constants = np.zeros((obs_user.shape[0], 1))

        month_indicator = []
        day_indicator = []
        week_indicator = []
        for idx, item_date_elem in enumerate(item_date):
            item_month = int(item_date_elem[-4:-2])
            # Month
            month_indicator.append(
                np.array([int(i == int(item_month) - 1) for i in range(12)]))
            # Day of Week
            day_indicator.append(
                np.array([int(i == item_dayofweek[idx] - 1) for i in range(7)]))
            # Week
            week_indicator.append(
                np.array([int(i == (item_week[idx] - 1) % 52) for i in range(52)]))
        
        month_indicator, week_indicator, day_indicator = map(np.array, (month_indicator, week_indicator, day_indicator))

        assert(month_indicator.shape[0] == item_date.shape[0])
        assert(day_indicator.shape[0] == item_date.shape[0])
        assert(week_indicator.shape[0] == item_date.shape[0])

        return (constants.astype(np.float64), obs_user.astype(np.float64), prices,
                item_onehot.astype(np.float64), month_indicator.astype(np.float64),
                day_indicator.astype(np.float64), week_indicator.astype(np.float64), A)

    def get_X_Y_all(self) -> Tuple[torch.Tensor]:
        complete_concat_x = np.hstack(
            [self.all_obs_user, self.all_prices, self.all_month_indicator, self.all_day_indicator])
        complete_concat_y = self.all_item_onehot
        assert(complete_concat_x.shape[0] == complete_concat_y.shape[0])
        return complete_concat_x, complete_concat_y

    def __getitem__(self, index):
        # item_date = self.data_arr[index][0]
        item_date = self.curr_training_data['date'].values[index]
        # item_week = self.data_arr[index][6]
        item_week = self.curr_training_data['wk'].values[index]
        # item_dayofweek = self.data_arr[index][7]
        item_dayofweek = self.curr_training_data['dayofweek'].values[index]
        # item_id = self.data_arr[index][1]
        item_id = self.curr_training_data['item_id'].values[index]
        # item_month = int(item_date[-4:-2])
        item_month = self.curr_training_data['date'].map(lambda x: int(str(x)[-4:-2])).values[index]
        if self.all_availabilities[index][np.argmax(self.all_item_onehot[index])] < 0.5:
            print('Purchased item not available\n')
            print(index, item_date, item_id, self.data_arr[index])

        # TODO(Tianyu): might need to modify what's returned here.
        return (np.hstack([self.all_constants[index], self.all_obs_user[index]]),
                self.all_prices[index],
                self.all_item_onehot[index],
                # np.float64(self.data_arr[index][4]),
                self.curr_training_data['category_id'].values,
                self.all_month_indicator[index],
                self.all_day_indicator[index],
                self.all_week_indicator[index],
                item_month,
                item_week,
                item_dayofweek,
                self.all_availabilities[index],
                np.int64(item_date),
                np.int64(item_id))

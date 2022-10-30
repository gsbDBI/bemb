import pdb
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils import data

sys.path.append("..")
torch.multiprocessing.set_sharing_strategy('file_system')
transaction_data = None


class Dataset_Train_Multitask(data.Dataset):
    def __init__(
            self,
            data_path: str,
            meta_path,
            item_stats_file,
            split: str,
            category=None,
            category_list=None,
            item_list=None,
            item_category=None,
            user_list=None,
            user_onehot=False,
            conditional_model=True,
            users_map=None,
    ):
        self.conditional_model = conditional_model
        self.category_we_care = category
        self.user_onehot = user_onehot
        self.users_map = users_map

        # Load sessional price dataset consisting of the price of item at each
        # session. A session consists of storeID + date.
        # Note that the `date` column is formatted as storeID + date(%Y%m%d).
        # The `date` column uniquely defines a session.
        columns = ['item_id', 'date', 'price']
        dtypes = [str, str, np.float64]
        dtypes = {a: b for a, b in zip(columns, dtypes)}
        self.item_sess_price_i = pd.read_csv(
            data_path + '/item_sess_price.tsv', sep='\t', dtype=dtypes,
            names=columns, header=None)
        self.item_sess_price_copy = pd.read_csv(
            data_path + '/item_sess_price.tsv', sep='\t', dtype=dtypes,
            names=columns, header=None)

        # Load itemID to categoryID mapping.
        self.itemGroup_i = pd.read_csv(
            data_path + '/itemGroup.tsv', sep='\t', header=None, dtype=str,
            names=['item_id', 'category_id'])

        # Load user observables.
        columns = ['user_id']
        # Include previously complied observable feature names.
        columns.extend(pd.read_csv(
            data_path + '/obsUserNames.tsv', sep='\t', header=None)[0])
        dtypes = [str] + [np.int64] * (len(columns) - 1)
        dtypes = {a: b for a, b in zip(columns, dtypes)}
        self.obsUser_i = pd.read_csv(
            data_path + "/obsUser.tsv", sep='\t', header=None, names=columns,
            dtype=dtypes)

        # Load information for sessions, recall that the `date` column in
        # dataframe uniquely defines a session.
        columns = ['date', 'wk', 'dayofweek', 'q', 'store']
        dtypes = [str] + [np.int64] * 3
        dtypes = {a: b for a, b in zip(columns, dtypes)}
        self.sess_days_i = pd.read_csv(
            data_path + '/sess_days.tsv', sep='\t', header=None,
            names=columns, dtype=dtypes)

        # Load the main purchasing history dataset.
        columns = ['user_id', 'item_id', 'date', 'num_purchase']
        dtypes = [str] * 3 + [np.int64]
        dtypes = {a: b for a, b in zip(columns, dtypes)}
        if (split != 'train'):
            assert (item_list is not None) and (user_list is not None), \
                'Please provide item list for test and validation'
        self.data_i = pd.read_csv(
            f'{data_path}/{split}.tsv', sep='\t', header=None, names=columns,
            dtype=dtypes)

        # Load availability of items.
        self.availability_data = pd.read_csv(
            data_path + '/availabilityList.tsv', sep='\t', header=None,
            names=['date', 'item_id'], dtype=str)

        # self.availability_date = self.availability_data[[
        #     'date']].drop_duplicates()
        # self.availability_date.columns = ['date']
        # self.availability_date = self.availability_date.sort_values(by=[
        #                                                             'date'])

        self.joint_training_data = self.data_i.set_index('item_id').join(
            self.itemGroup_i.set_index('item_id'), how='left'
        ).reset_index()
        self.curr_training_data = self.joint_training_data

        # Subset the specified category requested.
        if(self.category_we_care is not None):
            self.joint_training_data = self.joint_training_data.loc[
                self.joint_training_data['category_id'] == str(self.category_we_care)
            ]

        # Join trip (session) information to the main dataset.
        self.curr_training_data = self.curr_training_data.set_index('date') \
            .join(self.sess_days_i.set_index('date'), how='left').reset_index()

        self.curr_training_data = self.curr_training_data.sort_values(
            ['user_id', 'date'])

        # Build categoryID to itemID map.
        item_category_count = dict(
            self.joint_training_data
                .groupby('category_id')
                .apply(lambda x: list(np.unique(x['item_id'])))
        )
        assert sum(len(x) for x in item_category_count.values()) \
            == self.data_i['item_id'].nunique()

        # Convert required information to arrays.
        self.training_item_ids = self.curr_training_data['item_id']
        self.training_item_ids = np.array(self.training_item_ids)
        self.training_category_ids = self.curr_training_data['category_id']
        self.training_category_ids = np.array(self.training_category_ids)
        self.training_user_ids = list(
            self.curr_training_data['user_id'].drop_duplicates())
        self.train_size = self.training_item_ids.shape[0]

        # Build a list of unique values of each field, if not provided.
        self.category_list = category_list
        self.item_list = item_list
        self.item_category = item_category
        self.user_list = user_list
        if self.category_list is None:
            self.category_list = sorted(set(self.training_category_ids))
        if self.item_list is None:
            self.item_list = sorted(set(self.training_item_ids))
        if self.item_category is None:
            self.item_category = item_category_count
        if self.user_list is None:
            self.user_list = sorted(self.training_user_ids)

        self.num_categories = len(self.item_category.keys())
        # userId in dataset to consecutive integral IDs.
        self.user_ids_map = {elem: i for i, elem in enumerate(self.user_list)}
        self.availability_data = self.availability_data[self.availability_data['item_id'].isin(
            self.item_list)]  # One category each time
        # self.availability_data = self.availability_data.set_index(['date', 'item_id'])
        self.item_sess_price_copy = self.item_sess_price_copy[self.item_sess_price_copy['item_id'].isin(
            self.item_list)]  # One category each time
        self.report_stats(self.curr_training_data)
        # Count the frequency of each item.
        self.count_dict = {}
        _uniq = np.unique(self.training_item_ids, return_counts=True)
        self.count_dict = dict(zip(_uniq[0], _uniq[1]))
        # curr_training_data has columns=['date', 'item_id', 'user_id', 'num_purchase', 'category_id', 'wk', 'dayofweek', 'q', 'store']
        self.curr_training_data.sort_values(
            by=['date', 'user_id', 'item_id'], inplace=True)
        self.data_arr = np.array(self.curr_training_data)  # all integers.

        (self.all_constants, self.all_obs_user, self.user_onehots,
         self.all_prices, self.all_onehot, self.all_month_indicator,
         self.all_day_indicator, self.all_week_indicator,
         self.all_availabilities) = self.get_all_data()

        # Sanity check.
        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                print(f"{key}.shape={val.shape}")

        _, _ = self.get_X_Y_all()
        print("all data")

    def report_stats(self, df):
        item_ids = df['item_id']
        num_item_ids = len(sorted(set(np.array(item_ids))))
        print("num_item_ids: ", num_item_ids)

        user_ids = df['user_id']
        num_user_ids = len(sorted(set(np.array(user_ids))))
        print("num_user_ids: ", num_user_ids)

        date = df['date']
        num_date = len(sorted(set(np.array(date))))
        print("num_date: ", num_date)
        return None

    def __repr__(self):
        return "\n".join([
            f"{type(self)} object with",
            f"num unique item IDs: {self.curr_training_data['item_id'].nunique()}",
            f"num unique user IDs: {self.curr_training_data['user_id'].nunique()}",
            f"num unique category IDs: {self.curr_training_data['category_id'].nunique()}",
            f"store IDs: {list(self.curr_training_data['store'].unique())}",
            f"range of purchase date by store:",
            f"{self.curr_training_data.groupby('store')['date'].apply(lambda x: x.min() + ' to ' + x.max())}"
        ])

    def __len__(self):
        return self.curr_training_data.shape[0]

    def get_all_data(self):
        curr_train_data = self.curr_training_data
        curr_train_data = curr_train_data[curr_train_data['user_id'].isin(
            self.user_ids_map.keys())]
        curr_train_data = curr_train_data.sort_values(
            by=['user_id', 'date', 'category_id'])
        onehot = list()
        # Get trips, which is unique (user + store + date) tuples.
        # user_dates.shape[0] is the number of trips.
        user_dates = curr_train_data[['user_id', 'date', 'wk', 'dayofweek']].drop_duplicates()
        self.user_dates = user_dates
        users = list(user_dates['user_id'])
        user_indices = np.array([self.user_ids_map[user] for user in users])
        user_onehots = np.zeros(
            (user_indices.size, len(self.user_ids_map.keys())))
        user_onehots[np.arange(len(users)), user_indices] = 1

        curr_train_data = curr_train_data[[
            'user_id', 'date', 'category_id', 'item_id']]
        curr_train_data.reset_index(drop='True', inplace=True)
        self.long_form_train = curr_train_data

        all_availabilities = None
        self.availability_data['is_available'] = True
        all_availabilities = self.availability_data.pivot(
            index='date', columns='item_id', values='is_available')
        all_availabilities.fillna(False, inplace=True)
        # The pivot method should automatically sort columns and indices.
        assert list(all_availabilities.columns) == sorted(all_availabilities.columns)
        assert list(all_availabilities.index) == sorted(all_availabilities.index)
        assert all_availabilities.values.sum() == len(self.availability_data)

        train_wide = curr_train_data.groupby(
            ['user_id', 'date', 'category_id']
        )['item_id'].first().unstack('category_id')
        if self.conditional_model:
            for colname in self.category_list:
                if colname not in train_wide.columns:
                    train_wide[colname] = -1
        if not self.conditional_model:
            assert not train_wide.isnull().values.any(), \
                'Null training points in ' + train_wide
        else:
            # -1 indicates the user didn't buy this category at that trip.
            train_wide = train_wide.fillna(-1)

        train = train_wide[self.category_list]
        columns = list(train.columns)  # categories.
        train = np.array(train)

        # Convert itemID purchaseing history to ordinal encoded with consecutive integers.
        onehot = [
            [
                -1 if int(train[i, j]) < 0
                else self.item_category[columns[j]].index(train[i, j])
                for j in range(train.shape[1])
            ] for i in range(train.shape[0])
        ]
        onehot = np.array(onehot)
        assert onehot.shape == (len(train), len(self.category_list))

        num_items = len(self.item_list)
        self.item_sess_price_copy.sort_values(
            by=['date', 'item_id'], inplace=True)
        self.item_sess_price_copy.reset_index(drop='True', inplace=True)
        prices_wide = self.item_sess_price_copy.pivot(
            index='date', columns='item_id', values='price')
        prices_wide.fillna(0, inplace=True)
        prices = user_dates.set_index('date').join(prices_wide, how='left').reset_index()
        prices.sort_values(by=['user_id', 'date'], inplace=True)
        prices = prices.iloc[:, -num_items:]
        prices = prices[list(self.item_list)]
        prices = np.array(prices)
        assert prices.shape == (len(user_dates), len(self.item_list))

        obs_user = user_dates.set_index('user_id').join(
            self.obsUser_i.set_index('user_id'), how='left').reset_index()
        start_idx = -1 * (self.obsUser_i.shape[1] - 1)
        obs_user = np.array(obs_user)[:, start_idx:]
        constants = np.ones((obs_user.shape[0], 1))
        assert(obs_user.shape[0] == user_dates.shape[0])

        month_indicator = []
        day_indicator = []
        week_indicator = []
        for idx, date_elem in enumerate(user_dates.itertuples(index=False)):
            month = int(date_elem[1][-4:-2])
            dayofweek = date_elem[3]
            week = date_elem[2]

            # Month
            month_indicator.append(
                np.array([int(i == month - 1) for i in range(12)]))
            # Day of Week
            day_indicator.append(
                np.array([int(i == dayofweek - 1) for i in range(7)]))
            # Week
            week_indicator.append(
                np.array([int(i == (week - 1) % 52) for i in range(52)]))

        month_indicator = np.array(month_indicator)
        day_indicator = np.array(day_indicator)
        week_indicator = np.array(week_indicator)
        assert(month_indicator.shape[0] == user_dates.shape[0])
        assert(day_indicator.shape[0] == user_dates.shape[0])
        assert(week_indicator.shape[0] == user_dates.shape[0])
        return (constants.astype(np.float64), obs_user.astype(np.float64),
                user_onehots, prices, onehot.astype(np.int64),
                month_indicator.astype(np.float64),
                day_indicator.astype(np.float64),
                week_indicator.astype(np.float64),
                all_availabilities)

    def get_X_Y_all(self):
        if self.user_onehot:
            complete_concat_x = np.hstack([
                self.all_obs_user, self.user_onehots, self.all_prices,
                self.all_month_indicator, self.all_day_indicator])
        else:
            complete_concat_x = np.hstack([
                self.all_obs_user, self.all_prices,
                self.all_month_indicator, self.all_day_indicator])
        complete_concat_y = self.all_onehot
        assert(complete_concat_x.shape[0] == len(complete_concat_y))
        return complete_concat_x, complete_concat_y

    def get_X_Y_all_emb(self):
        assert self.user_onehot, "self.X_Y_all_emb only for user_onehot=True"
        complete_concat_x = np.hstack(
            [self.all_obs_user, self.all_prices, self.all_month_indicator, self.all_day_indicator])
        complete_concat_y = self.all_onehot
        assert(complete_concat_x.shape[0] == len(complete_concat_y))
        assert(complete_concat_x.shape[0] == self.user_onehots.shape[0])
        return complete_concat_x, self.user_onehots, complete_concat_y

    def __getitem__(self, index):
        item_date = self.data_arr[index][0]
        item_week = self.data_arr[index][6]
        item_dayofweek = self.data_arr[index][7]
        item_id = self.data_arr[index][1]
        item_month = int(item_date[-4:-2])
        if self.user_onehot:
            return np.hstack([self.all_constants[index], self.all_obs_user[index]]), self.all_prices[index], self.all_onehot[index], self.user_onehots[index], np.float64(self.data_arr[index][4]), self.all_month_indicator[index], self.all_day_indicator[index], self.all_week_indicator[index], item_month, item_week, item_dayofweek, None, np.int64(item_date), np.int64(item_id)
        else:
            return np.hstack([self.all_constants[index], self.all_obs_user[index]]), self.all_prices[index], self.all_onehot[index], np.float64(self.data_arr[index][4]), self.all_month_indicator[index], self.all_day_indicator[index], self.all_week_indicator[index], item_month, item_week, item_dayofweek, None, np.int64(item_date), np.int64(item_id)


class Dataset_Multitask_NN(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.Y[index])


class Dataset_Multitask_NN_emb(data.Dataset):
    def __init__(self, X, user_onehot, Y):
        self.X = X
        self.user_onehot = user_onehot
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.user_onehot[index], self.Y[index])

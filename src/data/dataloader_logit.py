# This is the data loader for multinomial logit, written by Yuan Yuan.
# Functions will be called in multinomial_logit_revision.py
# This is the cleanup version by Tianyu.
"""
Dataset for the first stage multinominal model.
"""
import numpy as np
from torch.utils import data
import pandas as pd
import math
import sys
sys.path.append("..")
import pdb


transaction_data = None

# Training Set


class Dataset_Train(data.Dataset):
    def __init__(self,
                 data_path,
                 meta_path,
                 category_we_care,
                 item_stats_file,
                 split,
                 item_list=None):
        # %% Load item prices at each (store, date) (called session).
        columns = ['item_id', 'date', 'price']
        dtypes = [str, str, np.float64]
        dtypes = {a: b for a, b in zip(columns, dtypes)}
        # header=None is for including the first line as header
        self.item_sess_price_i = pd.read_csv(
            data_path + "/item_sess_price.tsv", sep='\t', dtype=dtypes,
            names=columns, header=None)
        # header=None is for including the first line as header
        self.item_sess_price_copy = pd.read_csv(
            data_path + "/item_sess_price.tsv", sep='\t', dtype=dtypes,
            names=columns, header=None)

        # %% item ID to category ID mapping.
        columns = ['item_id', 'category_id']
        self.itemGroup_i = pd.read_csv(
            data_path + "/itemGroup.tsv", sep='\t', header=None, dtype=str,
            names=columns)

        # %% Load user observable features.
        columns = ['user_id']
        columns.extend(pd.read_csv(
            f'{data_path}/obsUserNames.tsv', sep='\t', header=None)[0])
        dtypes = [str] + [np.int64] * (len(columns) - 1)
        dtypes = {a: b for a, b in zip(columns, dtypes)}
        self.obsUser_i = pd.read_csv(
            data_path + '/obsUser.tsv', sep='\t', header=None, names=columns, dtype=dtypes)
        
        # %% Load calendar day information to sessions.
        columns = ['date', 'wk', 'dayofweek', 'q']
        dtypes = [str] + [np.int64] * 3
        dtypes = {a: b for a, b in zip(columns, dtypes)}
        self.sess_days_i = pd.read_csv(
            data_path + '/sess_days.tsv', sep='\t', header=None, names=columns, dtype=dtypes)
        
        # %% Load the main dataset of purchasing history.
        columns = ['user_id', 'item_id', 'date', 'num_purchase']
        dtypes = [str] * 3 + [np.int64]
        dtypes = {a: b for a, b in zip(columns, dtypes)}
        if (split != 'train'):
            assert item_list is not None,\
                'Please provide item list for test and validation.'
        self.data_i = pd.read_csv(
            f'{data_path}/{split}.tsv', sep='\t', header=None, names=columns,
            dtype=dtypes)
        
        # %%
        columns = ['item_no', 'store_cat_item_rank']
        dtypes = [str, np.int64]
        dtypes = {a: b for a, b in zip(columns, dtypes)}
        self.item_rank = pd.read_csv(
            "%s/%s" % (meta_path, item_stats_file), usecols=columns, dtype=dtypes)
        self.category_we_care = category_we_care
        self.item_rank.columns = ['item_id', 'item_rank']
        pdb.set_trace()

        # %%
        # TODO(Tianyu) What are these datasets, are we using them anyway? Comment out.
        self.category_stats = pd.read_csv(
            meta_path + '/category_stats.csv', dtype=str)
        columns = ['SubClassDescription', 'category_id']
        self.category_ids = pd.read_csv(
            meta_path + '/category_ids.csv', names=columns, dtype=str)
        self.itemGroup_no_dup = self.itemGroup_i.drop_duplicates(
            subset='category_id', keep="first")
        self.joint_category = self.category_stats.set_index(['SubClassDescription']).join(
            self.category_ids.set_index(['SubClassDescription']), how='left').reset_index()
        self.joint_category = self.itemGroup_no_dup.set_index(['category_id']).join(
            self.joint_category.set_index(['category_id']), how='left').reset_index()
        self.joint_category = self.joint_category.sort_values(
            by=['category_trips'], ascending=False)
        self.selected_category = np.array(
            self.joint_category['category_id'])[:40]

        # %% Load availability information.
        # Availability Info:
        columns = ['date', 'item_id']
        self.availability_data = pd.read_csv(
            data_path + "/availabilityList.tsv", sep='\t', header=None, names=columns, dtype=str)
        self.availability_date = self.availability_data[[
            'date']].drop_duplicates()
        self.availability_date.columns = ['date']
        self.availability_date = self.availability_date.sort_values(by=[
                                                                    'date'])

        # Add category into the main training set.
        self.joint_training_data = self.data_i.set_index(['item_id']).join(
            self.itemGroup_i.set_index(["item_id"]), how='left').reset_index()
        # Filter to get the category of interest.
        self.curr_training_data = self.joint_training_data.loc[
            self.joint_training_data['category_id'] == str(self.category_we_care)]  # One category each time

        self.curr_training_data = self.curr_training_data.set_index(['item_id']).join(
            self.item_rank.set_index(['item_id']), how='left').reset_index()  # Give item_rank column
        # self.curr_training_data = self.curr_training_data.loc[self.curr_training_data['item_rank'] <= 4] # Only keep top 5 ranked items
        # Add calendar day information associated with session.
        self.curr_training_data = self.curr_training_data.set_index(['date']).join(
            self.sess_days_i.set_index(['date']), how='left').reset_index()
        self.curr_training_data = self.curr_training_data.sort_values(
            ['item_id', 'date'])

        self.training_item_ids = self.curr_training_data['item_id']
        self.training_item_ids = np.array(self.training_item_ids)
        self.train_size = self.training_item_ids.shape[0]
        self.item_list = item_list
        # print("training_item_ids: ", self.training_item_ids, self.training_item_ids.shape)
        if self.item_list is None:
            self.item_list = sorted(set(self.training_item_ids))
        self.availability_data = self.availability_data[
            self.availability_data['item_id'].isin(self.item_list)]  # One category each time
        # self.availability_data = self.availability_data.set_index(['date', 'item_id'])
        self.item_sess_price_copy = self.item_sess_price_copy[
            self.item_sess_price_copy['item_id'].isin(self.item_list)]  # One category each time
        self.report_stats(self.curr_training_data)
        self.count_dict = {}
        for i in self.training_item_ids:
            if i not in self.count_dict:
                self.count_dict[i] = 1
            else:
                self.count_dict[i] += 1
        print("count_dict", self.count_dict)
        # print(self.curr_training_data.loc[self.curr_training_data['item_id'] == 100155992.0])
        # print(self.curr_training_data.loc[self.curr_training_data['item_id'] == 100155993.0])
        # print(self.curr_training_data.columns)
        self.curr_training_data.sort_values(
            by=['date', 'user_id', 'item_id'], inplace=True)
        self.data_arr = np.array(self.curr_training_data)
        self.all_constants, self.all_obs_user, self.all_prices, self.all_onehot, self.all_month_indicator, self.all_day_indicator, self.all_week_indicator, self.all_availabilities = self.get_all_data()
        _, _ = self.get_X_Y_all()
        # print(np.where(self.data_arr[:, 1] == 100155992.0))
        # print(np.where(self.data_arr[:, 1] == 100155993.0))
        # print(self.data_arr[847][1])
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

    def __len__(self):
        return self.curr_training_data.shape[0]

    def get_all_data(self):
        item = np.array(self.curr_training_data)
        # Get attributes and features in form of array.
        item_date = item[:, 0]  # ['date']
        item_id = item[:, 1]  # ['item_id']

        user_id = item[:, 2]  # ['user_id']
        item_category = item[:, 4]  # ['category_id']
        item_week = item[:, 6]  # ['wk']
        item_dayofweek = item[:, 7]  # ['dayofweek']
        # Onehot encoding of item purchased at each trip.
        onehot = np.array([
            np.array([1 if i == item_id_i else 0 for i in self.item_list])
            for item_id_i in item_id])
        self.item_sess_price_copy.sort_values(
            by=['date', 'item_id'], inplace=True)
        # print(self.item_sess_price_copy.dtypes)
        item_dates = self.item_sess_price_copy[['date', 'item_id']]
        # print(item_dates.dtypes)
        # input()
        # availability_data = self.availability_data.reset_index().sort_values(by=['date', 'item_id'])
        # print(availability_data.dtypes)
        # print(item_dates.dtypes)
        # input()
        
        # Add availability information.
        # _merge == 'both' indicates the ('item_id', 'date') presents in both
        # dataset, which implies item_id is available at this date.
        pdb.set_trace()
        # TODO(Tianyu): change to left join.
        item_dates['availability'] = 1

        outer_join = self.availability_data.merge(
            item_dates, how='outer', on=['item_id', 'date'], indicator=True)
        outer_join.rename(columns={"_merge": "availability"}, inplace=True)
        # print(outer_join)
        # print(outer_join.dtypes)
        outer_join["availability"] = outer_join["availability"].map(
            lambda x: 1 if x == "both" else 0).astype(np.int64)
        # print(outer_join.isnull().values.any())
        # print(outer_join.iloc[800])
        # print(item_dates.shape, outer_join.shape)
        # input()
        availabilities_wide = outer_join.pivot(
            index='date', columns='item_id', values='availability')
        availabilities_wide.fillna(0, inplace=True)
        # print(availabilities_wide[availabilities_wide.isnull().any(axis=1)])
        # input()
        num_items = len(self.item_list)
        availabilities = self.curr_training_data.set_index(
            'date').join(availabilities_wide, how='left').reset_index()
        availabilities.sort_values(by=["date", "user_id"], inplace=True)
        # print('\n')
        # print(availabilities.loc[availabilities['date'] == '301420160804']['100150731'])
        # print(availabilities.columns)
        # input()
        # print(availabilities.shape, onehot.shape)
        # print(availabilities.iloc[3368])
        # TODO(Tianyu): what's the point of this line?
        availabilities = availabilities.iloc[:, -num_items:]
        availabilities = availabilities[list(self.item_list)]
        # print(onehot[3368])

        # print(availabilities)
        # input()
        availabilities = np.array(availabilities)
        # print(availabilities[883:896])
        # input()
        assert(availabilities.shape[0] == item_date.shape[0])

        self.item_sess_price_copy.reset_index(drop='True', inplace=True)
        prices_wide = self.item_sess_price_copy.pivot(
            index='date', columns='item_id', values='price')
        prices_wide.fillna(0, inplace=True)
        # Merge on date to make the price vector collides the purchasing history.
        # Generate prices of all items at all trips.
        prices = self.curr_training_data.set_index(
            'date').join(prices_wide, how='left').reset_index()
        prices.sort_values(by=["date", "user_id"], inplace=True)
        # TODO(Tianyu): more safe to index using columns.
        prices = prices.iloc[:, -num_items:]
        prices = prices[list(self.item_list)]
        prices = np.array(prices)
        # print(np.argwhere(np.isnan(availabilities)))
        # print(np.argwhere(np.isnan(prices)))
        # input()
        assert(prices.shape[0] == item_date.shape[0])
        assert(prices.shape[1] == len(self.item_list))

        # Generate user observables at all trips.
        obs_user = self.curr_training_data.set_index('user_id').join(
            self.obsUser_i.set_index('user_id'), how='left').reset_index()
        # TODO(Tianyu): more safe to index using columns.
        start_idx = -1*(self.obsUser_i.shape[1] - 1)
        obs_user = np.array(obs_user)[:, start_idx:]
        constants = np.zeros((obs_user.shape[0], 1))
        assert(obs_user.shape[0] == item_date.shape[0])

        month_indicator = []
        day_indicator = []
        week_indicator = []
        for idx, item_date_elem in enumerate(item_date):
            item_month = int(item_date_elem[-4:-2])
            # Month
            month_indicator.append(
                np.array([int(i == int(item_month)-1) for i in range(12)]))
            # Day of Week
            day_indicator.append(
                np.array([int(i == item_dayofweek[idx]-1) for i in range(7)]))
            # Week
            week_indicator.append(
                np.array([int(i == (item_week[idx]-1) % 52) for i in range(52)]))

        month_indicator = np.array(month_indicator)
        day_indicator = np.array(day_indicator)
        week_indicator = np.array(week_indicator)
        assert(month_indicator.shape[0] == item_date.shape[0])
        assert(day_indicator.shape[0] == item_date.shape[0])
        assert(week_indicator.shape[0] == item_date.shape[0])
        return constants.astype(np.float64), obs_user.astype(np.float64), prices, onehot.astype(np.float64), month_indicator.astype(np.float64), day_indicator.astype(np.float64), week_indicator.astype(np.float64), availabilities
        # complete_concat_x = np.hstack([obs_user, prices, month_indicator, day_indicator, week_indicator])

    def get_X_Y_all(self):
        complete_concat_x = np.hstack(
            [self.all_obs_user, self.all_prices, self.all_month_indicator, self.all_day_indicator])
        complete_concat_y = self.all_onehot
        assert(complete_concat_x.shape[0] == complete_concat_y.shape[0])
        return complete_concat_x, complete_concat_y

    def __getitem__(self, index):
        item_date = self.data_arr[index][0]
        # print(index)
        # print(self.data_arr[index])
        item_week = self.data_arr[index][6]
        item_dayofweek = self.data_arr[index][7]
        item_id = self.data_arr[index][1]
        item_month = int(item_date[-4:-2])
        if self.all_availabilities[index][np.argmax(self.all_onehot[index])] < 0.5:
            pirnt("Purchased item not available\n")
            print(index, item_date, item_id, self.data_arr[index])
            input()

        # print(item_id)
        # print(type(item_week), type(item_dayofweek))
        return np.hstack([self.all_constants[index], self.all_obs_user[index]]), self.all_prices[index], self.all_onehot[index], np.float64(self.data_arr[index][4]), self.all_month_indicator[index], self.all_day_indicator[index], self.all_week_indicator[index], item_month, item_week, item_dayofweek, self.all_availabilities[index], np.int64(item_date), np.int64(item_id)

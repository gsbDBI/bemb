# This script generates a systhic consumer choice dataset. 
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings


# def generate_users(num_users: int, num_user_feat: int) -> torch.Tensor:
#     all_feat = list()
#     # # generate gender.
#     # gender = torch.Tensor(np.random.randint(2, size=(num_users)))
#     # all_feat.append(gender)
#     # # generate age.
#     # age = torch.Tensor(np.random.randn(num_users) * 5 + 20).round()
#     # all_feat.append(age)
#     # # generate income.
#     # # income = torch.Tensor(np.exp(np.random.randn(num_users) * 3 + 5))
#     # income = torch.Tensor(np.random.uniform(size=(num_users)) * 3 + 100)
#     # all_feat.append(income)
    
#     if num_user_feat > num_users:
#         warnings.warn('More user features required than users, might cause identifiability issues.')
    
#     low = - num_user_feat // 2
#     mu_lst = np.random.choice(np.arange(low, low + num_user_feat), size=num_user_feat, replace=False)
#     sigma_lst = np.random.choice(np.arange(1, num_user_feat + 1), size=num_user_feat, replace=False)
    
#     for mu, sigma in zip(mu_lst, sigma_lst):
#         val = np.random.randn(num_users) * sigma + mu
#         all_feat.append(torch.Tensor(val))
    
#     X = torch.stack(all_feat, dim=1)
#     assert X.shape == (num_users, num_user_feat)
#     return X


# def generate_items(num_items: int) -> torch.Tensor:
#     all_feat = list()
#     price = torch.Tensor(np.exp(np.random.randn(num_items) + 1))
#     all_feat.append(price)
#     # quality is somehow positively correlated with price.
#     quality = torch.Tensor(np.exp(np.random.randn(num_items)) * price.numpy()).round()
#     all_feat.append(quality)
#     X = torch.stack(all_feat, dim=1)
#     return X


def generate_gaussian_features(num_classes: int, num_features: int) -> torch.Tensor:
    all_feat = list()

    low = - num_features // 2
    mu_lst = np.random.choice(np.arange(low, low + num_features), size=num_features, replace=False)
    sigma_lst = np.random.choice(np.arange(1, num_features + 1), size=num_features, replace=False)
    
    for mu, sigma in zip(mu_lst, sigma_lst):
        val = np.random.randn(num_classes) * sigma + mu
        all_feat.append(torch.Tensor(val))
    
    X = torch.stack(all_feat, dim=1)
    assert X.shape == (num_classes, num_features)
    return X


def generate_Xy(user_sess, user_feat, item_feat):
    num_sessions = len(user_sess)
    num_users, num_user_features = user_feat.shape
    num_items, num_item_features = item_feat.shape
    # cast user features to all items.
    X_user = user_feat[user_sess, :].view(num_sessions, 1, num_user_features).expand(-1, num_items, -1)
    # cast item features to all sessions. 
    X_item = item_feat.view(1, num_items, num_item_features).expand(num_sessions, -1, -1)

    X = {'u': X_user, 'i': X_item}
    
    coef = dict()
    # user-specific features have coefficients varying across items. (type = item_full)
    num_coefs = num_items * num_user_features
    coef['u'] = torch.arange(-num_coefs // 2, -num_coefs // 2 + num_coefs).view(1, num_items, num_user_features)
    coef['u'] = coef['u'].expand(num_sessions, -1, -1)
    # item-specific features varies across users (type = user).
    num_coefs = num_users * num_item_features
    c = torch.arange(-num_coefs // 2, -num_coefs // 2 + num_coefs).view(num_users, num_item_features)
    c = c[user_sess, :].view(num_sessions, 1, num_item_features).expand(-1, num_items, -1)
    coef['i'] = c
 
    utility = (X['u'] * coef['u']).sum(dim=-1) + (X['i'] * coef['i']).sum(dim=-1)
 
    noise = np.random.gumbel(size=(num_sessions, num_items)) * float(utility.std() / 10)
    utility += noise
    choice = torch.argmax(utility, dim=1)
    return X, choice


def generate_dataset(num_users: int,
                     num_user_features: int,
                     num_items: int,
                     num_item_features: int,
                     num_sessions: int):
    user_feat = generate_gaussian_features(num_users, num_user_features)
    item_feat = generate_gaussian_features(num_items, num_item_features)
    # which user is involved in each session.
    user_sess = torch.Tensor(np.random.randint(low=0, high=num_users, size=(num_sessions,))).long()

    X, y = generate_Xy(user_sess, user_feat, item_feat)
    user_onehot = torch.zeros(num_sessions, num_users)
    user_onehot[torch.arange(num_sessions), user_sess] = 1
    return X, user_onehot, y


if __name__ == '__main__':
    X, user_onehot, y = generate_dataset(num_users=10,
                                         num_user_features=128,
                                         num_items=4,
                                         num_item_features=64,
                                         num_sessions=10000)

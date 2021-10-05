# Data Mangement in `DeepChoice`

The `deepchoice` package offers a comperhensive `deepchoice.data.ChoiceDataset` object for easier data mangement. 

**NOTE**: we use the term *feature* and *observable* exchangably through this documentation.

## Build `ChoiceDataset` from Tensors

**TODO**: add link to jupyter-noteook.

```python
num_users = 10
num_user_features = 128
num_items = 4
num_item_features = 64
num_sessions = 10000
```

### 1. Create Observables.

Currently, the `deepchoice` package supports five types of observables, each type of observables depends on different elements from `{user, item, session}`.

| (Vary by)  User | Item | Session | Variable Name | Raw Tensor Shape                                             | Definition                                                   |
| --------------- | ---- | ------- | ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| No              | No   | No      | `intercept`   | N/A, the model automatically creates `torch.ones(num_sessions, num_items, 1)`  for intercepts. Do not create intercept dummies in the dataset. | N/A                                                          |
| Yes             | No   | No      | `user_xxx`    | `(num_users, *)`                                             | dataset of user-specific features. E.g., user's income.      |
| No              | Yes  | No      | `item_xxx`    | `(num_items, *)`                                             | dataset of item specific features that is constant over time/sessions. E.g., the quality of a product. |
| No              | No   | Yes     | `session_xxx` | `(num_sessions, *)`                                          | dataset of session-specific features. E.g., the day of week when session was observed, modelling the day of week capturing seasonal effects. |
| Yes             | Yes  | No      | `taste_xxx`   | `(num_users, num_items, *)`                                  | dataset of features depending on both user and item. E.g., the interaction between user's income level and an indicator of luxury good. |
| Yes             | No   | Yes     | Not Allowed   | N/A                                                          | N/A                                                          |
| No              | Yes  | Yes     | `price_xxx`   | `(num_sessions, num_items, *)`                               | item-specific features that varying across session/time as well. E.g., the price of item. **NOTE**: researchers should add user-item-session-specific features, such as the ratio of user's income and itme's price, here. |
| Yes             | Yes  | Yes     | Now Allowed   | N/A                                                          | N/A                                                          |

We firstly create some arbitrary tensors for `{user, item, session, taste, price}` observables. All tensors are constructed randomly here for pedagogical purpose.

```python
user_obs = torch.randn(num_users, 128)  # generate 128 features for each user.
item_obs = torch.randn(num_items, 64)  # generate 64 features for each user.
session_obs = torch.randn(num_sessions, 234)  # generate 234 features for each user.
taste_obs = torch.randn(num_users, num_items, 567)  # generate 567 features for each user.
price_obs = torch.randn(num_sessions, num_items, 12)  # generate 12 features for each user.
```

### 2. Create Label, User Onehot and Item Availabilities.

* `label` is a long tensor with the item ID chosen in each session.
* `user_onehot` is a binary tensor indicating which user is making the decision in that 

```python
label = torch.LongTensor(np.random.choice(num_items, size=num_sessions))

user_onehot = torch.zeros(num_sessions, num_users)
user_idx = torch.LongTensor(np.random.choice(num_users, size=num_sessions))
user_onehot[torch.arange(num_sessions), user_idx] = 1

item_availability = torch.ones(num_sessions, num_items).bool()
```



## Convert Long-Format to `ChoiceDataset`



## Convert Wide-Format to `ChoiceDataset`


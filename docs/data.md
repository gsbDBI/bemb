# Data Mangement in `DeepChoice`

The `deepchoice` package offers a comperhensive `deepchoice.data.ChoiceDataset` object for easier data mangement. 

**NOTE**: we use the term *feature* and *observable* exchangably through this documentation.



## Design Principle

The ultimate goal of `deepchoice` is to estimate $$U_{uit}$$, the utility for user $$u$$ to choose item $$i$$ in session $$t$$ . Session is indexed by $$t$$ because the most common approach is defining session by time, however, we do allow for more general definition of sessions such as the intereaction between location and time. The dataset consists of the following components, they behave differently during mini-batching.

### Raw Observables

Currently, the `deepchoice` package supports five types of observables, each type of observables depends on different elements from `{user, item, session}`.

| (Vary by)  User (u) | Item (i) | Session (t) | Variable Name Prefix | Raw Tensor Shape                                             | Definition                                                   |
| ------------------- | -------- | ----------- | -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| No                  | No       | No          | `intercept`          | N/A, the model automatically creates `torch.ones(num_sessions, num_items, 1)`  for intercepts. Do not create intercept dummies in the dataset. | N/A                                                          |
| Yes                 | No       | No          | `user_xxx`           | `(num_users, *)`                                             | dataset of user-specific features. E.g., user's income.      |
| No                  | Yes      | No          | `item_xxx`           | `(num_items, *)`                                             | dataset of item specific features that is constant over time/sessions. E.g., the quality of a product. |
| No                  | No       | Yes         | `session_xxx`        | `(num_sessions, *)`                                          | dataset of session-specific features. E.g., the day of week when session was observed, modelling the day of week capturing seasonal effects. |
| Yes                 | Yes      | No          | `taste_xxx`          | `(num_users, num_items, *)`                                  | dataset of features depending on both user and item. E.g., the interaction between user's income level and an indicator of luxury good. |
| Yes                 | No       | Yes         | Not Allowed          | N/A                                                          | N/A                                                          |
| No                  | Yes      | Yes         | `price_xxx`          | `(num_sessions, num_items, *)`                               | item-specific features that varying across session/time as well. E.g., the price of item. **NOTE**: researchers should add user-item-session-specific features, such as the ratio of user's income and itme's price, here. |
| Yes                 | Yes      | Yes         | Now Allowed          | N/A                                                          | N/A                                                          |

### Item Availablity

Item availability is a boolean-valued `price` feature with shape `(num_sessions, num_items)`. **NOTE**: even through `item_availablity`  has the prefix `item_`, it is a reserved attribute and will **not** be considered as an item-specific variable.

### Index Tensors

We support two types of index tensors, `user_index` and `session_index`, there is **no** `item_index` because we need to compute utilites for all items anyways for likelihoods and inclusive values. While computing $U_{uit}$  for a list of $(u, t)$, `{user, session}_index` are used to identify the corresponding user-specific features, session/price-sepcific features, as well as item availability.  While conducting mini-batching, the sampler does sampling over index-tensors. For example, to compute $$U_{uit} = \beta X^{user:obs}_{u} + \gamma X^{price:obs}_{it}$$ for $$L$$ pairs of user and session,  $$(u_\ell, t_\ell)_{\ell=1}^L$$, index tensors are constructed as the following:

```python
user_index = [u1, u2, ..., uL]
session_index = [t1, t2, ..., tL]
```

Using those tensor indices, we can obtain the corresponding $$X^{user:obs}_{u}$$ and $$X^{price:obs}_{it}$$ by `user_obs[user_index, :]` and `price_obs[session_index, :, :]`.

Naturally, suppose $$L$$ is too large and we wish to split $$L$$ into smaller chunks (batches) with $$B$$  pairs of $$(u_\ell, t_\ell)_{\ell=1}^B$$ in each batch. For simplicity, if we want observations 1,3,5,7,9 from the dataset to be in the mini-batch, we simply subset index tensors  as  `user_index = user_index[[1,3,5,7,9]]` and `session_index = session_index[[1,3,5,7,9]]`.

### Label

The `dataset.label`  is a tensor with the same length as the dataset, it stores the ID of the item bought. While consutrcuting batches using, for example,  observations 1,3,5,7,9 from the dataset, we take `label = label[[1,3,5,7,9]]`.

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


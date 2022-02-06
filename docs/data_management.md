<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Data Management
The `torch_choice` and `bemb` packages share the `ChoiceDataset` data structure for managing choice histories.
The `ChoiceDataset` is an instance of the PyTorch dataset object, which allows for easy training with mini-batch sampling.

We provided a Jupyter notebook for this tutorial as well, you can find the notebook located at `/tutorials/data_management.ipynb` in the

## Setup the Choice-Dataset
The BEMB model was initially designed for predicting consumers’ purchasing choices from the supermarket purchase dataset, we use the same setup in this tutorial. However, one can easily adopt the `ChoiceDataset` data structure to other use cases.
Since there is a natural notion of **users** in the modelling problem, from now on we will call individuals in the dataset **users** and you, the reader, **researchers**.

Since we will be using PyTorch to train our model, we represent their identities with integer values.

### Items and Categories
We begin with notations and essential component of the consumer choice modelling problem.
Suppose there are $$I$$ **items** indexed by $$i \in \{1,2,\dots,I\}$$ under our consideration.
Further, the researcher can optionally partition the set items into $$C$$ **categories** indexed by $$c \in \{1,2,\dots,C\}$$. Let $$I_c$$ denote the collection of items in category $$c$$, it is easy to verify that
$$
\bigcup_{c \in \{1, 2, \dots, C\}} I_c = \{1, 2, \dots I\}
$$
If the researcher does not wish to model different categories differently, the researcher can simply put all items in one single category: $$I_1 = \{1, 2, \dots I\}$$.

### Users
Let $$B$$ denote the number of **purchasing records** in the dataset (i.e., number of rows of the dataset). Each row $$b \in \{1,2,\dots, B\}$$ corresponds to a purchase record (i.e., *who* bought *what* at *where and when*). Each row is naturally associated with an **user** indexed by $$u \in \{1,2,\dots,U\}$$ (*who*) and a bought item $$i$$ (*what*).
Overall, each row (i.e., purchasing record) in the dataset is characterized by a user-session-item tuple $$(u, s, i)$$.
When there are multiple items bought by the same user in the same session, there will be multiple rows in the dataset with the same $$(u, s)$$.

### Sessions
Our data structure encompasses *where and when* using a notion called **session** indexed by $$s \in \{1,2,\dots, For example, of session $$s$$ is the date of purchase.
Another example is that we have the purchase record from different stores, the session $$s$$ can be defined as a pair of *(date, store)* instead.
If the researcher does not wish to handle records from different sessions differently, the researcher can assign the same session ID to all rows of the dataset.

### Item Availability
It is not necessarily that all items are available for purchasing everyday, items can get out-of-stock in particular sessions. To handle these cases, the researcher can optionally provide a boolean tensor $$\in \{\texttt{True}, \texttt{False}\}^{S\times I}$$ to indicate which items are available for purchasing in each session.
While predicting the purchase probabilities, the model sets the probability for these unavailable items to zero and normalizes probabilities among available items.
If the item availability is not provided, the model assumes all items are available in all sessions.

To summarize, the `ChoiceDataset` is expecting the following keyword argument while being constructed:
1. `label` $$\in \{1,2,\dots,I\}^B$$ : the ID of bought item for each purchasing record (*what*).
2. `user_index` $$\in \{1,2,\dots,U\}^B$$: the ID of the corresponding user (shopper)for each purchasing record (*who*).
3. `session_index` $$\in \{1,2,\dots,S\}^B$$: the corresponding session of each purchasing record (*where and when*).
4. `item_availability` $$\in \{\texttt{True}, \texttt{False}\}^{S\times I}$$  identifies the availability of items in each session, the model will ignore unavailable items while making prediction.

### Observables
#### Basic Usage
Optionally, the researcher can incorporate observables of, for example, users and items. Currently, the package support the following types of observables, where $$K_{...}$ $denote the number of observables.

1. `user_obs` $$\in \mathbb{R}^{U\times K_{user}}$$
2. `item_obs` $$\in \mathbb{R}^{I\times K_{item}}$$
3. `session_obs` $$\in \mathbb{R}^{S \times K_{session}}$$
4. `price_obs` $$\in \mathbb{R}^{S \times I \times K_{price}}$$, price observables are values depending on **both** session and item.

The researcher should supply them with as appropriate keyword arguments while constructing the `ChoiceDataset` object.

#### Advanced Usage: Additional Observables
In some cases, the researcher may wish to handle different parts of `user_obs` (or other observable tensors) differently.
For example, the researcher wishes to model the utility for user $$u$$ to purchase item $$i$$ in session $$s$$ as the following:
$$
U_{usi} = \beta_{i} X^{(u)}_{user\ income} + \gamma X^{(u)}_{user\ market\ membership}
$$
The coefficient for user income is item-specific so that it captures the nature of the product (i.e., a luxury or an essential good). Additionally, the utility representation admits an user market membership becomes shoppers with active memberships tend to purchase more, and the coefficient of this term is constant across all items.
As we will cover later in the modelling section, we need to supply two user observable tensors in this case for the model to build coefficient with different levels of variations (i.e., item-specific coefficients versus constant coefficients). In this case, the researcher needs to supply two tensors `user_income` and `user_market_membership` as keyword arguments to the `ChoiceDataset` constructor.
The `ChoiceDataset` handles multiple user/item/session/price observables internally, for example, every keyword arguments passed into `ChoiceDataset` with name starting with `item_` (except for the reserved `item_availability`) will be treated as item observable tensors. All keywords with names starting `user_`, `session_` and `price_` (except for reserved names like `user_index` and `session_index` mentioned above) will be interpreted as user/session/price observable tensors.

## Toy Example
Thi section provides a toy example of `ChoiceDataset`.
Suppose we have a dataset of purchase records from two stores (Store A and B) on two dates (Sep 16 and 17), both stores offered {apple, banana, orange} and there are three users,{Amy, Ben, Charlie} in this dataset.
The table below provides an example of `ChoiceDataset` with length 5, 3 users, 3 items, and 4 sessions (each session is a store-date pair).

| user_index | session_index       | label  |
| ---------- | ------------------- | ------ |
| Amy        | Sep-17-2021-Store-A | banana |
| Ben        | Sep-17-2021-Store-B | apple  |
| Ben        | Sep-16-2021-Store-A | orange |
| Charlie    | Sep-16-2021-Store-B | apple  |
| Charlie    | Sep-16-2021-Store-B | orange |

For demonstration purpose, the example dataset has `user_index`, `session_index` and `label` as strings, however, as mentioned before, they should be encoded integer values for PyTorch compatibility.
One can easily convert them to integers using `sklearn.preprocessing.LabelEncoder` to get `user_index=[0,1,1,2,2]`, `session_index=[0,1,2,3,3]`, and `label=[0,1,2,1,2]` for the example provided above.

Suppose we believe people's purchasing decision depends on nutrition levels of these fruits, suppose apple has the highest nutrition level and banana has the lowest one, we can add `item_obs=[1.5, 12.0, 3.3]` (recall the integer encoding of fruits based on the `label` variable above. These numbers were arbitrary) as keyword argument to the `ChoiceDataset`.

## Coding Example
This section provides an introduction to the functionality of `ChoiceDataset` and the `JointDataset` wrapper, which chains multiple `ChoiceDataset` together. `JointDataset` will be particularly useful when we are training `NestedLogitModel`s.

### Creating `ChoiceDataset` Object
#### Step 1: Generate some random purchase records and observables
We will be creating a randomly generated dataset with 10000 purchase records from 10 users, 4 items and 500 sessions.
```python
import numpy as np
import torch
from torch_choice.data import ChoiceDataset, JointDataset

num_users = 10
num_items = 4
num_sessions = 500
length_of_dataset = 10000
```

The first step is to randomly generate the purchase records using the following code. For simplicity, we assume all items are available in all sessions.
```python
label = torch.LongTensor(np.random.choice(num_items, size=length_of_dataset))  # what was bought
user_index = torch.LongTensor(np.random.choice(num_users, size=length_of_dataset))  # who bought this
session_index = torch.LongTensor(np.random.choice(num_sessions, size=length_of_dataset))  # when and where

# assume all items are available in all sessions.
item_availability = torch.ones(num_sessions, num_items).bool()
```

We then generate random observable tensors for users, items, sessions and price observables, the size of observables of each type (i.e., the last dimension in the shape) is arbitrarily chosen.
```python
user_obs = torch.randn(num_users, 128)  # generate 128 features for each user, e.g., race, gender.
item_obs = torch.randn(num_items, 64)  # generate 64 features for each user, e.g., quality.
session_obs = torch.randn(num_sessions, 10)  # generate 10 features for each session, e.g., weekday indicator.
price_obs = torch.randn(num_sessions, num_items, 12)  # generate 12 features for each session user pair, e.g., the budget of that user at the shopping day.
```

#### Step 2: Initialize the `ChoiceDataset`.
You can now construct a `ChoiceDataset` for randomly generated purchase records with the following code, which manage all information for you.
```python
dataset = ChoiceDataset(
    label=label,
    user_index=user_index,
    session_index=session_index,
    item_availability=item_availability,
    user_obs=user_obs,
    item_obs=item_obs,
    session_obs=session_obs,
    price_obs=price_obs)
```

## Functionality of the `ChoiceDataset`
The `ChoiceDataset` object provides helper functions for the researcher to inspect the dataset.

### `print(dataset)` and `dataset.__str__`
The command `print(dataset)` will provide a quick overview of shapes of tensors included in the object as well as where the dataset is located (i.e., host memory or GPU memory).
```python
print(dataset)

# output:
# ChoiceDataset(label=[10000], user_index=[10000], session_index=[10000], item_availability=[500, 4], observable_prefix=[5], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], price_obs=[500, 4, 12], device=cpu)
```

### `dataset.num_{users, items, sessions}`
You can use the `num_{users, items, sessions}` attribute to obtain the number of users, items, and sessions, they are determined automatically from the `{user, item, session}_obs` tensors provided while initializing the dataset object.
For example:
```python
print(f'{dataset.num_users=:}')
print(f'{dataset.num_items=:}')
print(f'{dataset.num_sessions=:}')
print(f'{len(dataset)=:}')

# output:
# dataset.num_users=10
# dataset.num_items=4
# dataset.num_sessions=500
# len(dataset)=10000
```

### `dataset.clone()`
The `ChoiceDataset` offers a `clone` method allow you to make copy of the dataset, you can modify the cloned dataset arbitrarily without changing the original dataset.
```python
print(dataset.label[:10])
dataset_cloned = dataset.clone()
dataset_cloned.label = 99 * torch.ones(num_sessions)
print(dataset_cloned.label[:10])
print(dataset.label[:10])  # does not change the original dataset.

# output
# tensor([1, 3, 0, 2, 1, 3, 0, 3, 1, 1])
# tensor([99., 99., 99., 99., 99., 99., 99., 99., 99., 99.])
# tensor([1, 3, 0, 2, 1, 3, 0, 3, 1, 1])
```

### `dataset.to('cuda')` and `dataset._check_device_consistency()`.
One key advantage of the `torch_choice` and `bemb` is their compatibility with GPUs, you can easily move tensors in a `ChoiceDataset` object between host memory (i.e., cpu memory) and device memory (i.e., GPU memory) using `dataset.to()` method.
Please note that the following code runs only if your machine has a compatible GPU and GPU-compatible version of PyTorch installed.
```python
# move to device
print(f'{dataset.device=:}')
print(f'{dataset.label.device=:}')
print(f'{dataset.user_index.device=:}')
print(f'{dataset.session_index.device=:}')

dataset = dataset.to('cuda')

print(f'{dataset.device=:}')
print(f'{dataset.label.device=:}')
print(f'{dataset.user_index.device=:}')
print(f'{dataset.session_index.device=:}')

# output
# dataset.device=cpu
# dataset.label.device=cpu
# dataset.user_index.device=cpu
# dataset.session_index.device=cpu
# dataset.device=cuda:0
# dataset.label.device=cuda:0
# dataset.user_index.device=cuda:0
# dataset.session_index.device=cuda:0
```

Similarly, one can move data to host-memory using `dataset.to('cpu')`.
The dataset also provides a `dataset._check_device_consistency()` method to check if all tensors are on the same device.
If we only move the `label` to cpu without moving other tensors, this will result in an error message.
```python
dataset.label = dataset.label.to('cpu')
dataset._check_device_consistency()

# will raise error message.
```

### Subset method
One can use `dataset[indices]` with `indices` as an integer-valued tensor or array to get the corresponding rows of the dataset.
The example code block below queries the 6256-th, 4119-th, 453-th, 5520-th, and 1877-th row of the dataset object.
The `label`, `user_index`, `session_index` of the resulted subset will be different from the original dataset, but other tensors
will be the same.
```python
dataset = dataset.to('cpu')
indices = torch.Tensor(np.random.choice(len(dataset), size=5, replace=False)).long()
print(indices)
subset = dataset[indices]
print(dataset)
print(subset)

assert torch.all(dataset.x_dict['price_obs'][indices, :, :] == subset.x_dict['price_obs'])
assert torch.all(dataset.label[indices] == subset.label)

# output
# tensor([6256, 4119,  453, 5520, 1877])
# ChoiceDataset(label=[10000], user_index=[10000], session_index=[10000], item_availability=[500, 4], observable_prefix=[5], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], price_obs=[500, 4, 12], device=cpu)
# ChoiceDataset(label=[5], user_index=[5], session_index=[5], item_availability=[500, 4], observable_prefix=[5], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], price_obs=[500, 4, 12], device=cpu)
```
The subset method internally creates a copy of the datasets so that any modification applied on the subset will **not** be reflected on the original dataset.
The researcher can feel free to do in-place modification to the subset.
```python
print(subset.label)  # output: tensor([3, 1, 1, 0, 2])
print(dataset.label[indices])  # output: tensor([3, 1, 1, 0, 2])

subset.label += 1  # modifying the batch does not change the original dataset.

print(subset.label)  # output: tensor([4, 2, 2, 1, 3]), updated.
print(dataset.label[indices])  # output: tensor([3, 1, 1, 0, 2]), will NOT change.
```
Similarly, you can do in-place modification on observable tensors of the subset without affecting the original dataset as well.
```python
print(subset.item_obs[0, 0])
print(dataset.item_obs[0, 0])
subset.item_obs += 1
print(subset.item_obs[0, 0])
print(dataset.item_obs[0, 0])

# output;
# tensor(-0.4046)
# tensor(-0.4046)
# tensor(0.5954)
# tensor(-0.4046)
```

## Chaining Multiple Datasets: `JointDataset` Examples
```python
dataset1 = dataset.clone()
dataset2 = dataset.clone()
joint_dataset = JointDataset(the_dataset=dataset1, another_dataset=dataset2)
joint_dataset

# output
# JointDataset with 2 sub-datasets: (
# 	the_dataset: ChoiceDataset(label=[10000], user_index=[10000], session_index=[10000], item_availability=[500, 4], observable_prefix=[5], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], price_obs=[500, 4, 12], device=cpu)
# 	another_dataset: ChoiceDataset(label=[10000], user_index=[10000], session_index=[10000], item_availability=[500, 4], observable_prefix=[5], user_obs=[10, 128], item_obs=[4, 64], session_obs=[500, 10], price_obs=[500, 4, 12], device=cpu)
# )
```

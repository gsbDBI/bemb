# Inference using the BEMB model

This tutorial covers methods for post-estimation inference. After training the BEMB model, it is useful to have a more detailed look at predictions from the model. Functionalities covered in this tutorial allows you to make prediction on new datasets. Methods here (i.e., `forward()` and `predict_proba()`) are versatile and offering both predicted probabilities and predicted utilities.

Author: Tianyu Du

Date: Aug. 8, 2022

Update: Aug. 10, 2022


```python
import sys
from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch

# we use the dataset simulation method from unit tests.
sys.path.append('../../tests')
import simulate_choice_dataset
import torch
from bemb.model import LitBEMBFlex
from torch_choice.data import ChoiceDataset
```

    /Users/tianyudu/miniforge3/envs/ml/lib/python3.9/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.1
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"


## Generate Simulated Datasets
We will use the simulated dataset in [this tutorial](https://gsbdbi.github.io/bemb/bemb_obs2prior_simulation/) for demonstration purpose.

The simulated dataset is divided into train (80%), validation (10%), and test (10%) subsets automatically.

Moreover, the simulated dataset includes 50-dimensional user observables and item observables. [This tutorial](https://gsbdbi.github.io/bemb/bemb_obs2prior_simulation/) mentioned definitions of these observables but you don't need to know them for the purpose of this tutorial.


```python
num_users = 1500
num_items = 50
data_size = 10000

# split into three train, validation and test datasets.
dataset_list = simulate_choice_dataset.simulate_dataset(num_users=num_users, num_items=num_items, data_size=data_size)
dataset_list
```

    No `session_index` is provided, assume each choice instance is in its own session.





    [ChoiceDataset(label=[], item_index=[8000], user_index=[8000], session_index=[8000], item_availability=[], user_obs=[1500, 50], item_obs=[50, 50], device=cpu),
     ChoiceDataset(label=[], item_index=[1000], user_index=[1000], session_index=[1000], item_availability=[], user_obs=[1500, 50], item_obs=[50, 50], device=cpu),
     ChoiceDataset(label=[], item_index=[1000], user_index=[1000], session_index=[1000], item_availability=[], user_obs=[1500, 50], item_obs=[50, 50], device=cpu)]



## Construct and Train the Model

Here we will be using a rather simple model with two sets of parameters, an user latent $\theta_u \in \mathbb{R}^{10}$ for each of the 1,500 users, and an item latent $\alpha_i \in \mathbb{R}^{10}$ for each of the 50 items.

**Note**: The behavior of `forward()` and `predict_proba()` methods depends on the model setup (i.e., whether `pred_item` or not).

**Note**: The `LitBEMBFlex` object is a class wrapping the actual model with training loops, to access the core model encompassed, we use `bemb.model` (see the `return` line of `train_model()`).


```python
def train_model(pred_item: bool):
    bemb = LitBEMBFlex(
        learning_rate=0.03,  # set the learning rate, feel free to play with different levels.
        pred_item=pred_item,
        num_seeds=32,  # number of Monte Carlo samples for estimating the ELBO.
        utility_formula='theta_user * alpha_item',  # the utility formula.
        # tell the model some necessary information about the setup.
        num_users=num_users,
        num_items=num_items,
        num_user_obs=dataset_list[0].user_obs.shape[1],
        num_item_obs=dataset_list[0].item_obs.shape[1],
        # whether to turn on obs2prior for each parameter.
        obs2prior_dict={'theta_user': True, 'alpha_item': True},
        # the dimension of latents, since the utility is an inner product of theta and alpha, they should have
        # the same dimension.
        coef_dim_dict={'theta_user': 10, 'alpha_item': 10}
    )

    # use GPU if available.
    if torch.cuda.is_available():
        bemb = bemb.to('cuda')
        
    # use the provided run helper to train the model.
    # we set batch size to be 5% of the data size, and train the model for 10 epochs.
    # there would be 20*10=200 gradient update steps in total.
    bemb = bemb.fit_model(dataset_list, batch_size=len(dataset_list[0]) // 20, num_epochs=10, num_workers=0)
    # The `LitBEMBFlex` object is a class wrapping the actual model with training loops, to access the core model encompassed, we use `bemb.model`.
    return bemb.model
```

## The `forward()` Function

The `forward()` function is the main workhorse for inference, please see the doc-string of `forward()` function for definitions of its arguments.


```python
def forward(self, batch: ChoiceDataset,
            return_type: str,
            return_scope: str,
            deterministic: bool = True,
            sample_dict: Optional[Dict[str, torch.Tensor]] = None,
            num_seeds: Optional[int] = None
            ) -> torch.Tensor:
    """A combined method for inference with the model.

    Args:
        batch (ChoiceDataset): batch data containing choice information.
        return_type (str): either 'log_prob' or 'utility'.
            'log_prob': return the log-probability (by within-category log-softmax) for items
            'utility': return the utility value of items.
        return_scope (str): either 'item_index' or 'all_items'.
            'item_index': for each observation i, return log-prob/utility for the chosen item batch.item_index[i] only.
            'all_items': for each observation i, return log-prob/utility for all items.
        deterministic (bool, optional):
            True: expectations of parameter variational distributions are used for inference.
            False: the user needs to supply a dictionary of sampled parameters for inference.
            Defaults to True.
        sample_dict (Optional[Dict[str, torch.Tensor]], optional): sampled parameters for inference task.
            This is not needed when `deterministic` is True.
            When `deterministic` is False, the user can supply a `sample_dict`. If `sample_dict` is not provided,
            this method will create `num_seeds` samples.
            Defaults to None.
        num_seeds (Optional[int]): the number of random samples of parameters to construct. This is only required
            if `deterministic` is False (i.e., stochastic mode) and `sample_dict` is not provided.
            Defaults to None.
    Returns:
        torch.Tensor: a tensor of log-probabilities or utilities, depending on `return_type`.
            The shape of the returned tensor depends on `return_scope` and `deterministic`.
            -------------------------------------------------------------------------
            | `return_scope` | `deterministic` |         Output shape               |
            -------------------------------------------------------------------------
            |   'item_index` |      True       | (len(batch),)                      |
            -------------------------------------------------------------------------
            |   'all_items'  |      True       | (len(batch), num_items)            |
            -------------------------------------------------------------------------
            |   'item_index' |      False      | (num_seeds, len(batch))            |
            -------------------------------------------------------------------------
            |   'all_items'  |      False      | (num_seeds, len(batch), num_items) |
            -------------------------------------------------------------------------
    """
    # function body omitted.
    return None
```

With our simple model, for the $k$-th purchasing record (observation) in the dataset, suppose `dataset.user_index[k] = ` $u(k)$ and `dataset.user_index[k] = ` $i(k)$.
Suppose there are $K$ such observations in the dataset.

After training the model, the model now contain fitted values of $\theta$'s and $\alpha$'s, inference predictions will be based on these parameters. 

Our simple model calculates $$U_{u(k) \ell} = \theta_{u(k)}^\top \alpha_\ell$$ for every possible item $\ell$ including the chosen $i(k)$. This is called user $u(k)$'s *utility* from buying item $\ell$.

As mentioned before, interpretations of the `forward()` function are slightly different depending on whether `pred_item == True`.

### Summary Table for `forward()` Method


| `pred_item`| `return_scope` | `return_type`  | Output Shape | Output Tensor|
| :---: | :----: | :---: | :---: | :---: |
| `True`  | `item_index`    |  `utility`  | `(len(batch),)`| $\left[\theta_{u(1)}^\top \alpha_{i(1)}, \theta_{u(2)}^\top \alpha_{i(2)}, \dots, \theta_{u(K)}^\top \alpha_{i(K)}\right]$|
| `True`  | `item_index`    |  `log_prob`  | `(len(batch),)`| $\left[\log\left(\frac{\exp(\theta_{u(1)}^\top \alpha_{i(1)})}{\sum_{\ell \in \text{category of } i(1)} \exp(\theta_{u(1)}^\top \alpha_{\ell}) }\right), \log\left(\frac{\exp(\theta_{u(2)}^\top \alpha_{i(2)})}{\sum_{\ell \in \text{category of } i(2)} \exp(\theta_{u(2)}^\top \alpha_{\ell}) }\right), \dots, \log\left(\frac{\exp(\theta_{u(K)}^\top \alpha_{i(K)})}{\sum_{\ell \in \text{category of } i(K)} \exp(\theta_{u(K)}^\top \alpha_{\ell}) }\right)\right]$|
| `True`  | `all_items`    |  `utility`  | `(len(batch), num_items)`| $\begin{bmatrix} \theta_{u(1)}^\top \alpha_{1}, \theta_{u(1)}^\top \alpha_{2}, \dots, \theta_{u(1)}^\top \alpha_{num\_items} \\ \theta_{u(2)}^\top \alpha_{1}, \theta_{u(2)}^\top \alpha_{2}, \dots, \theta_{u(2)}^\top \alpha_{num\_items} \\ \vdots \\ \theta_{u(K)}^\top \alpha_{1}, \theta_{u(K)}^\top \alpha_{2}, \dots, \theta_{u(K)}^\top \alpha_{num\_items} \end{bmatrix}$|
| `True`  | `all_items`    |  `log_prob`  | `(len(batch), num_items)`| $\begin{bmatrix} \log\left(\frac{\exp(\theta_{u(1)}^\top \alpha_{1})}{\sum_{\ell \in \text{category of } 1} \exp(\theta_{u(1)}^\top \alpha_{\ell}) }\right), \log\left(\frac{\exp(\theta_{u(1)}^\top \alpha_{2})}{\sum_{\ell \in \text{category of } 2} \exp(\theta_{u(1)}^\top \alpha_{\ell}) }\right), \dots, \log\left(\frac{\exp(\theta_{u(1)}^\top \alpha_{num\_items})}{\sum_{\ell \in \text{category of } num\_items} \exp(\theta_{u(1)}^\top \alpha_{\ell}) }\right) \\ \log\left(\frac{\exp(\theta_{u(2)}^\top \alpha_{1})}{\sum_{\ell \in \text{category of } 1} \exp(\theta_{u(2)}^\top \alpha_{\ell}) }\right), \log\left(\frac{\exp(\theta_{u(2)}^\top \alpha_{2})}{\sum_{\ell \in \text{category of } 2} \exp(\theta_{u(2)}^\top \alpha_{\ell}) }\right), \dots, \log\left(\frac{\exp(\theta_{u(2)}^\top \alpha_{num\_items})}{\sum_{\ell \in \text{category of } num\_items} \exp(\theta_{u(2)}^\top \alpha_{\ell}) }\right) \\ \vdots \\ \log\left(\frac{\exp(\theta_{u(K)}^\top \alpha_1)}{\sum_{\ell \in \text{category of } 1} \exp(\theta_{u(K)}^\top \alpha_{\ell}) }\right), \log\left(\frac{\exp(\theta_{u(K)}^\top \alpha_2)}{\sum_{\ell \in \text{category of } 2} \exp(\theta_{u(K)}^\top \alpha_{\ell}) }\right), \dots, \log\left(\frac{\exp(\theta_{u(K)}^\top \alpha_{num\_items})}{\sum_{\ell \in \text{category of } num\_items} \exp(\theta_{u(K)}^\top \alpha_{\ell}) }\right) \end{bmatrix}$|
| `False`  | `item_index`    |  `utility`  | `(len(batch),)`| $\left[\theta_{u(1)}^\top \alpha_{i(1)}, \theta_{u(2)}^\top \alpha_{i(2)}, \dots, \theta_{u(K)}^\top \alpha_{i(K)}\right]$|
| `False`  | `item_index`    |  `log_prob`  | `(len(batch),)`| $\left[y_1 \log\left(\sigma\left(\theta_{u(1)}^\top \alpha_{i(1)}\right)\right) + (1-y_1) \log\left(1-\sigma\left(\theta_{u(1)}^\top \alpha_{i(1)}\right)\right), y_2 \log\left(\sigma\left(\theta_{u(2)}^\top \alpha_{i(2)}\right)\right) + (1-y_2) \log\left(1-\sigma\left(\theta_{u(2)}^\top \alpha_{i(2)}\right)\right), \dots, y_K \log\left(\sigma\left(\theta_{u(K)}^\top \alpha_{i(K)}\right)\right) + (1-y_K) \log\left(1-\sigma\left(\theta_{u(K)}^\top \alpha_{i(K)}\right)\right)\right]$|
| `False`  | `all_items`    |  `utility`  | `(len(batch), num_items)`| $\begin{bmatrix} \theta_{u(1)}^\top \alpha_{1}, \theta_{u(1)}^\top \alpha_{2}, \dots, \theta_{u(1)}^\top \alpha_{num\_items} \\ \theta_{u(2)}^\top \alpha_{1}, \theta_{u(2)}^\top \alpha_{2}, \dots, \theta_{u(2)}^\top \alpha_{num\_items} \\ \vdots \\ \theta_{u(K)}^\top \alpha_{1}, \theta_{u(K)}^\top \alpha_{2}, \dots, \theta_{u(K)}^\top \alpha_{num\_items} \end{bmatrix}$|
| `False`  | `all_items`    |  `log_prob`  | `(len(batch), num_items)`| $\begin{bmatrix} y_1 \log\left(\sigma\left(\theta_{u(1)}^\top \alpha_{1}\right)\right) + (1-y_1) \log\left(1-\sigma\left(\theta_{u(1)}^\top \alpha_{1}\right)\right), y_1 \log\left(\sigma\left(\theta_{u(1)}^\top \alpha_{2}\right)\right) + (1-y_1) \log\left(1-\sigma\left(\theta_{u(1)}^\top \alpha_{2}\right)\right) , \dots, y_1 \log\left(\sigma\left(\theta_{u(1)}^\top \alpha_{num\_items}\right)\right) + (1-y_1) \log\left(1-\sigma\left(\theta_{u(1)}^\top \alpha_{num\_items}\right)\right) \\ y_2 \log\left(\sigma\left(\theta_{u(2)}^\top \alpha_{1}\right)\right) + (1-y_2) \log\left(1-\sigma\left(\theta_{u(2)}^\top \alpha_{1}\right)\right), y_2 \log\left(\sigma\left(\theta_{u(2)}^\top \alpha_{2}\right)\right) + (1-y_2) \log\left(1-\sigma\left(\theta_{u(2)}^\top \alpha_{2}\right)\right) , \dots, y_2 \log\left(\sigma\left(\theta_{u(2)}^\top \alpha_{num\_items}\right)\right) + (1-y_2) \log\left(1-\sigma\left(\theta_{u(2)}^\top \alpha_{num\_items}\right)\right) \\ \vdots \\ y_K \log\left(\sigma\left(\theta_{u(K)}^\top \alpha_{1}\right)\right) + (1-y_K) \log\left(1-\sigma\left(\theta_{u(K)}^\top \alpha_{1}\right)\right), y_K \log\left(\sigma\left(\theta_{u(K)}^\top \alpha_{2}\right)\right) + (1-y_K) \log\left(1-\sigma\left(\theta_{u(K)}^\top \alpha_{2}\right)\right) , \dots, y_K \log\left(\sigma\left(\theta_{u(K)}^\top \alpha_{num\_items}\right)\right) + (1-y_K) \log\left(1-\sigma\left(\theta_{u(K)}^\top \alpha_{num\_items}\right)\right) \end{bmatrix}$|

### Predicting Item Index (`pred_item == True`)
In this case, the model aims to predict *which item user $u(k)$ would purchase*.

Let's get a copy of the simple model.


```python
%%capture
# use %%capture to hide cumbersome logs of training.
model = train_model(pred_item=True)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    
      | Name  | Type     | Params
    -----------------------------------
    0 | model | BEMBFlex | 33.0 K
    -----------------------------------
    33.0 K    Trainable params
    0         Non-trainable params
    33.0 K    Total params
    0.132     Total estimated model params size (MB)


Let's then make predicting using the test set (the last entry in `dataset_list`).


```python
batch = dataset_list[-1]
```


Recall that, for each observation $k$, suppose $u(k)$ is the corresponding user and item $i(k)$ was chosen. With learned parameters $\theta_u$ and $\alpha_i$, the model first calculates $$U_{u(k) \ell} = \theta_{u(k)}^\top \alpha_\ell$$ for every possible item $\ell$ including the chosen $i(k)$. This is called user $u(k)$'s *utility* from buying item $\ell$.

Then, the predicted probability for user $u(k)$ to purchase item $i$ is:
$$\frac{\exp(\theta_{u(1)}^\top \alpha_i)}{\sum_{\ell \in \text{category of } i} \exp(\theta_{u(1)}^\top \alpha_{\ell}) }$$

The denominator was normalizing using all items belonging to the category of item $i$, however, in this example, we don't consider items' categories (i.e., assuming all of them belong to the same category). The predicted probability becomes:
$$\frac{\exp(\theta_{u(1)}^\top \alpha_i)}{\sum_{\ell=1}^{num\_items} \exp(\theta_{u(1)}^\top \alpha_{\ell}) }$$

The `forward()` function allows you to get predicted probabilities (actually, log-probability for numerical stability) for all items. You would need to specify 
* `return_scope='all_items'`
* `return_type='log_prob'`

As we expected, the shape of `log_prob_all_items` is $K$ by num-items, so that `log_prob_all_items[k, i]` denotes the predicted log-probability for user $u(k)$ to choose item $i$ in the context of the $k$-th observation.

Formally, the returned tensor is:

$$\begin{bmatrix} \log\left(\frac{\exp(\theta_{u(1)}^\top \alpha_{1})}{\sum_{\ell \in \text{category of } 1} \exp(\theta_{u(1)}^\top \alpha_{\ell}) }\right), \log\left(\frac{\exp(\theta_{u(1)}^\top \alpha_{2})}{\sum_{\ell \in \text{category of } 2} \exp(\theta_{u(1)}^\top \alpha_{\ell}) }\right), \dots, \log\left(\frac{\exp(\theta_{u(1)}^\top \alpha_{num\_items})}{\sum_{\ell \in \text{category of } num\_items} \exp(\theta_{u(1)}^\top \alpha_{\ell}) }\right) \\ \log\left(\frac{\exp(\theta_{u(2)}^\top \alpha_{1})}{\sum_{\ell \in \text{category of } 1} \exp(\theta_{u(2)}^\top \alpha_{\ell}) }\right), \log\left(\frac{\exp(\theta_{u(2)}^\top \alpha_{2})}{\sum_{\ell \in \text{category of } 2} \exp(\theta_{u(2)}^\top \alpha_{\ell}) }\right), \dots, \log\left(\frac{\exp(\theta_{u(2)}^\top \alpha_{num\_items})}{\sum_{\ell \in \text{category of } num\_items} \exp(\theta_{u(2)}^\top \alpha_{\ell}) }\right) \\ \vdots \\ \log\left(\frac{\exp(\theta_{u(K)}^\top \alpha_1)}{\sum_{\ell \in \text{category of } 1} \exp(\theta_{u(K)}^\top \alpha_{\ell}) }\right), \log\left(\frac{\exp(\theta_{u(K)}^\top \alpha_2)}{\sum_{\ell \in \text{category of } 2} \exp(\theta_{u(K)}^\top \alpha_{\ell}) }\right), \dots, \log\left(\frac{\exp(\theta_{u(K)}^\top \alpha_{num\_items})}{\sum_{\ell \in \text{category of } num\_items} \exp(\theta_{u(K)}^\top \alpha_{\ell}) }\right) \end{bmatrix}$$


```python
log_prob_all_items = model.forward(batch, return_scope='all_items', return_type='log_prob')
print(f"{log_prob_all_items.shape=:}")
```

    log_prob_all_items.shape=torch.Size([1000, 50])


The predicted probabilities in each row of `log_prob_all_items` should (approximately) sum to one.


```python
print(log_prob_all_items.exp())
print(f"{log_prob_all_items.exp().sum(dim=1).max()=:}")
print(f"{log_prob_all_items.exp().sum(dim=1).min()=:}")
```

    tensor([[0.0419, 0.0089, 0.0380,  ..., 0.0266, 0.0098, 0.0428],
            [0.0104, 0.0393, 0.0197,  ..., 0.0196, 0.0078, 0.0346],
            [0.0284, 0.0262, 0.0207,  ..., 0.0041, 0.0049, 0.0062],
            ...,
            [0.0043, 0.0255, 0.0065,  ..., 0.0147, 0.0056, 0.0263],
            [0.0101, 0.0215, 0.0091,  ..., 0.0168, 0.0462, 0.0052],
            [0.0191, 0.0283, 0.0014,  ..., 0.0049, 0.0254, 0.0192]],
           grad_fn=<ExpBackward0>)
    log_prob_all_items.exp().sum(dim=1).max()=1.0000003576278687
    log_prob_all_items.exp().sum(dim=1).min()=0.9999995827674866


Sometimes we want raw values of utilities of each user $u(k)$ from purchasing each item $i$. To achieve this, we specify:
* `return_type='utility'`

In this case, the returned tensor is:

$$\begin{bmatrix} \theta_{u(1)}^\top \alpha_{1}, \theta_{u(1)}^\top \alpha_{2}, \dots, \theta_{u(1)}^\top \alpha_{num\_items} \\ \theta_{u(2)}^\top \alpha_{1}, \theta_{u(2)}^\top \alpha_{2}, \dots, \theta_{u(2)}^\top \alpha_{num\_items} \\ \vdots \\ \theta_{u(K)}^\top \alpha_{1}, \theta_{u(K)}^\top \alpha_{2}, \dots, \theta_{u(K)}^\top \alpha_{num\_items} \end{bmatrix}$$


```python
utility_all_items = model.forward(batch, return_scope='all_items', return_type='utility')
print(f"{utility_all_items.shape=:}")
```

    utility_all_items.shape=torch.Size([1000, 50])


Note that there is no guarantee on the row-sum of this tensor.


```python
print(utility_all_items)
```

    tensor([[ 0.7441, -0.8005,  0.6468,  ...,  0.2911, -0.7060,  0.7649],
            [-0.6786,  0.6514, -0.0381,  ..., -0.0436, -0.9664,  0.5246],
            [ 0.5187,  0.4397,  0.2012,  ..., -1.4056, -1.2460, -1.0009],
            ...,
            [-0.9111,  0.8633, -0.4985,  ...,  0.3130, -0.6555,  0.8928],
            [-0.2202,  0.5378, -0.3241,  ...,  0.2927,  1.3049, -0.8828],
            [ 0.4644,  0.8605, -2.1168,  ..., -0.8846,  0.7492,  0.4722]],
           grad_fn=<SqueezeBackward1>)


In some other cases, such as while computing the log-likelihood to assess the goodness-of-fit for the entire model, we only care about user $u(k)$'s utility/log-probability for item $i(k)$ that she/he actually bought.

Specifying 
* `return_scope = 'item_index`
calculates these values much faster on large datasets and/or we have many categories of items compared to the `log_prob_all_items[torch.arange(len(batch)), batch.item_index]` operation.


```python
log_prob_item_index = model.forward(batch, return_scope='item_index', return_type='log_prob')
print(f"{log_prob_item_index.shape=:}")
```

    log_prob_item_index.shape=torch.Size([1000])



```python
torch.all(log_prob_item_index == log_prob_all_items[torch.arange(len(batch)), batch.item_index])
```




    tensor(True)




```python
utility_item_index = model.forward(batch, return_scope='item_index', return_type='utility')
print(f"{utility_item_index.shape=:}")
torch.all(utility_item_index == utility_all_items[torch.arange(len(batch)), batch.item_index])
```

    utility_item_index.shape=torch.Size([1000])





    tensor(True)



### Predicting Binary Labels (`pred_item == False`)
We have a label $y_k \in \{0, 1\}$ for each observation $k$, the model can either output the (1) raw utility or (2) the predicted $\hat{P}(y_k = 1)$ defined as $\sigma(U)$, where $\sigma(x) = \frac{1}{1 + \exp(-x)}$.

**Notes**
1. the return shape does not depend on the `return_type`, please compare exact expressions to see the difference.
2. the `pred_item` variable was specified while initializing the model (see above), `return_scope` and `return_type` are supplied while calling `forward()`.


```python
for dataset in dataset_list:
    # assign some trivial labels.
    dataset.label = torch.Tensor(dataset.user_index >= (num_users // 2)).long()
```


```python
%%capture
# use %%capture to hide cumbersome logs of training.
model = train_model(pred_item=False)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    
      | Name  | Type     | Params
    -----------------------------------
    0 | model | BEMBFlex | 33.0 K
    -----------------------------------
    33.0 K    Trainable params
    0         Non-trainable params
    33.0 K    Total params
    0.132     Total estimated model params size (MB)



```python
batch = dataset_list[-1]
```

With `return_scope='item_index', return_type='utility'`, the `forward()` method returns
$$\left[\theta_{u(1)}^\top \alpha_{i(1)}, \theta_{u(2)}^\top \alpha_{i(2)}, \dots, \theta_{u(K)}^\top \alpha_{i(K)}\right].$$
This is indeed the same as the case wth `pred_item = True` above.

Therefore, `utility_item_index[k]` below is the utility of user $u(k)$ from purchasing the item $i(k)$ in the $k$-th observation.  

With `pred_item = False`, each observation $k$ is now associated with a label $y_k \in \{0, 1\}$. With `return_type='log_prob'`, the `forward()` function returns the predicted probability of the label $y_k$.

Specifically, the model will predict the probability of the positive class (i.e., $y_k = 1$), condition on latent of user $u(k)$ and item $i(k)$ as
$$
P(y_k=1 | \theta_{u(k)}, \alpha_{i(k)}) = \sigma(\theta_{u(k)}^\top \alpha_{i(k)}) = \frac{1}{1 + \exp\left(-\theta_{u(K)}^\top \alpha_{i(K)}\right)} \in [0, 1]
$$
Therefore, the $k$-th entry of the returned tensor (i.e., `log_prob_item_index[k]` below) is $\log \sigma(\theta_{u(k)}^\top \alpha_{i(k)})$ if $y_k = 1$ and $\log \left(1 - \sigma(\theta_{u(k)}^\top \alpha_{i(k)})\right)$ if $y_k = 0$.
This can be equivalently written as
$$\left[y_1 \log\left(\sigma\left(\theta_{u(1)}^\top \alpha_{i(1)}\right)\right) + (1-y_1) \log\left(1-\sigma\left(\theta_{u(1)}^\top \alpha_{i(1)}\right)\right), y_2 \log\left(\sigma\left(\theta_{u(2)}^\top \alpha_{i(2)}\right)\right) + (1-y_2) \log\left(1-\sigma\left(\theta_{u(2)}^\top \alpha_{i(2)}\right)\right), \dots, y_K \log\left(\sigma\left(\theta_{u(K)}^\top \alpha_{i(K)}\right)\right) + (1-y_K) \log\left(1-\sigma\left(\theta_{u(K)}^\top \alpha_{i(K)}\right)\right)\right]$$


```python
utility_item_index = model.forward(batch, return_scope='item_index', return_type='utility')
print(f'{utility_item_index.shape=:}')
log_prob_item_index = model.forward(batch, return_scope='item_index', return_type='log_prob')
print(f'{log_prob_item_index.shape=:}')
```

    utility_item_index.shape=torch.Size([1000])
    log_prob_item_index.shape=torch.Size([1000])


But, what if I want $P(y_k=1 | \theta_{u(k)}, \alpha_{i(k)})$ for every observation $k$? The solution is straightforward, we simply take the $\sigma(\cdot)$ transformation of utilities returned by `utility_all_items = model.forward(batch, return_scope='item_index', return_type='utility')`.

Please note that when `return_type = 'utility'`, the dataset (`batch`) doesn't need to have a `label` attribute! For example, you might have `label` on your training dataset, but you want to conduct inference on a new dataset without known labels. You can simply create a `ChoiceDataset` object without the `label` attribute and use the following method to draw inference on it (e.g., get predicted probabilities of positive classes).

Here is an example:


```python
# make a copy of the batch.
batch_copy = batch.clone()
# manually delete the label attribute, with return_type = 'utility', we don't need this.
del batch_copy.label
```


```python
A = model.forward(batch_copy, return_scope='item_index', return_type='utility')
prob_positive_class = 1 / (1 + torch.exp(-A))
```

Recall that `log_prob_item_index = model.forward(batch, return_scope='item_index', return_type='log_prob')` reports the predicted log-probability of the actual label $y_k$, namely $\log P(y_k | \theta_{u(k)}, \alpha_{i(k)})$.
Therefore, the relationship between `log_prob_item_index` we computed before and the `prob_positive_class` tensor we just computed is
$$
\texttt{log\_prob\_item\_index} = \log P(y_k | \theta_{u(k)}, \alpha_{i(k)}) = \begin{cases}
\log P(y_k=1 | \theta_{u(k)}, \alpha_{i(k)}) &\text{ if } y_k = 1 \\
\log \left(1 - P(y_k=1 | \theta_{u(k)}, \alpha_{i(k)})\right) &\text{ if } y_k = 0 \\
\end{cases} 
= \begin{cases}
\log \texttt{prob\_positive\_class} &\text{ if } y_k = 1 \\
\log \left(1 - \texttt{prob\_positive\_class}\right) &\text{ if } y_k = 0 \\
\end{cases}
$$

Let confirm the relationship now.


```python
y = batch.label
log_prob_item_index_from_alternative_method = y * torch.log(prob_positive_class) + (1 - y) * torch.log(1 - prob_positive_class)
torch.all(log_prob_item_index == log_prob_item_index_from_alternative_method)
```




    tensor(True)



Since different items have different latent $\alpha_\ell$, the predicted probability of $y_k = 1$ depends on the item chosen.
By setting `return_scope='all_items'`, the `forward()` method returns $\theta_{u(k)}^\top \alpha_{\ell}$ and $\log P(y_k | \theta_{u(k)}, \alpha_{\ell})$ for all items $\ell \in \{1, 2, \dots, num\_items\}$.

Formally, the `utility_all_items` tensors contains:
$$
\begin{bmatrix} \theta_{u(1)}^\top \alpha_{1}, \theta_{u(1)}^\top \alpha_{2}, \dots, \theta_{u(1)}^\top \alpha_{num\_items} \\ \theta_{u(2)}^\top \alpha_{1}, \theta_{u(2)}^\top \alpha_{2}, \dots, \theta_{u(2)}^\top \alpha_{num\_items} \\ \vdots \\ \theta_{u(K)}^\top \alpha_{1}, \theta_{u(K)}^\top \alpha_{2}, \dots, \theta_{u(K)}^\top \alpha_{num\_items} \end{bmatrix}
$$

and the `log_prob_all_items` tensors contains:
$$\begin{bmatrix} y_1 \log\left(\sigma\left(\theta_{u(1)}^\top \alpha_{1}\right)\right) + (1-y_1) \log\left(1-\sigma\left(\theta_{u(1)}^\top \alpha_{1}\right)\right), y_1 \log\left(\sigma\left(\theta_{u(1)}^\top \alpha_{2}\right)\right) + (1-y_1) \log\left(1-\sigma\left(\theta_{u(1)}^\top \alpha_{2}\right)\right) , \dots, y_1 \log\left(\sigma\left(\theta_{u(1)}^\top \alpha_{num\_items}\right)\right) + (1-y_1) \log\left(1-\sigma\left(\theta_{u(1)}^\top \alpha_{num\_items}\right)\right) \\ y_2 \log\left(\sigma\left(\theta_{u(2)}^\top \alpha_{1}\right)\right) + (1-y_2) \log\left(1-\sigma\left(\theta_{u(2)}^\top \alpha_{1}\right)\right), y_2 \log\left(\sigma\left(\theta_{u(2)}^\top \alpha_{2}\right)\right) + (1-y_2) \log\left(1-\sigma\left(\theta_{u(2)}^\top \alpha_{2}\right)\right) , \dots, y_2 \log\left(\sigma\left(\theta_{u(2)}^\top \alpha_{num\_items}\right)\right) + (1-y_2) \log\left(1-\sigma\left(\theta_{u(2)}^\top \alpha_{num\_items}\right)\right) \\ \vdots \\ y_K \log\left(\sigma\left(\theta_{u(K)}^\top \alpha_{1}\right)\right) + (1-y_K) \log\left(1-\sigma\left(\theta_{u(K)}^\top \alpha_{1}\right)\right), y_K \log\left(\sigma\left(\theta_{u(K)}^\top \alpha_{2}\right)\right) + (1-y_K) \log\left(1-\sigma\left(\theta_{u(K)}^\top \alpha_{2}\right)\right) , \dots, y_K \log\left(\sigma\left(\theta_{u(K)}^\top \alpha_{num\_items}\right)\right) + (1-y_K) \log\left(1-\sigma\left(\theta_{u(K)}^\top \alpha_{num\_items}\right)\right) \end{bmatrix}$$


```python
print(f'{model.pred_item=:}')
utility_all_items = model.forward(batch, return_scope='all_items', return_type='utility')
print(f'{utility_all_items.shape=:}')
log_prob_all_items = model.forward(batch, return_scope='all_items', return_type='log_prob')
print(f'{log_prob_all_items.shape=:}')
```

    model.pred_item=False
    utility_all_items.shape=torch.Size([1000, 50])
    Using the new version...
    log_prob_all_items.shape=torch.Size([1000, 50])


Let check these tensors are consistent with the `return_scope='item_index'` case:


```python
print(torch.all(utility_all_items[torch.arange(len(batch)), batch.item_index] == utility_item_index))
print(torch.all(log_prob_all_items[torch.arange(len(batch)), batch.item_index] == log_prob_item_index))
```

    tensor(True)
    tensor(True)


If you want the predicted probability for the positive class $y_k = 1$, then you can simply apply sigmoid function $\sigma()$ to the `utility_all_items` tensor.

### The `deterministic` Option
By default, the `forward()` function has keyword argument `deterministic = True`. In this case, the model uses the means of fitted variational distributions of $\theta$ and $\alpha$ to compute utilities and log-probabilities.

One can specify `forward(deterministic=False, num_seeds=<XXX>)`, the model will firstly sample `num_seeds` copies of $\theta$ and $\alpha$ from their variational distributions. For each copy, the model calculated utility/log-probability as described above.

Therefore, with the same `pred_item`, `return_scope`, and `return_type`, the returned tensor has shape `(num_seeds, <the shape described in the table above>)`.

For example, for a model initialized with `pred_item=False`,

* `forward(batch, return_scope='all_items', return_type='utility', deterministic=True)` returns shape `(len(batch), num_items)` as mentioned above.
* However, `forward(batch, return_scope='all_items', return_type='utility', deterministic=False, num_seeds=32)` returns shape `(32, len(batch), num_items)`.

Here is an actual example (with `pred_item = False`):


```python
deterministic = model.forward(batch, return_scope='item_index', return_type='utility', deterministic=True)
random = model.forward(batch, return_scope='item_index', return_type='utility', deterministic=False, num_seeds=128)
print(f"{deterministic.shape=:}")
print(f"{random.shape=:}")
# the mean absolute difference between deterministic estimation and random estimation.
torch.mean(torch.abs(deterministic - random.mean(dim=0)))
```

    deterministic.shape=torch.Size([1000])
    random.shape=torch.Size([128, 1000])





    tensor(0.0433, grad_fn=<MeanBackward0>)




```python
deterministic = model.forward(batch, return_scope='all_items', return_type='utility', deterministic=True)
random = model.forward(batch, return_scope='all_items', return_type='utility', deterministic=False, num_seeds=128)
print(f"{deterministic.shape=:}")
print(f"{random.shape=:}")
# the mean absolute difference between deterministic estimation and random estimation.
torch.mean(torch.abs(deterministic - random.mean(dim=0)))
```

    deterministic.shape=torch.Size([1000, 50])
    random.shape=torch.Size([128, 1000, 50])





    tensor(0.0443, grad_fn=<MeanBackward0>)




```python
deterministic = model.forward(batch, return_scope='all_items', return_type='log_prob', deterministic=True)
random = model.forward(batch, return_scope='all_items', return_type='log_prob', deterministic=False, num_seeds=128)
print(f"{deterministic.shape=:}")
print(f"{random.shape=:}")
# the mean absolute difference between deterministic estimation and random estimation.
torch.mean(torch.abs(deterministic - random.mean(dim=0)))
```

    Using the new version...
    Using the new version...
    deterministic.shape=torch.Size([1000, 50])
    random.shape=torch.Size([128, 1000, 50])





    tensor(0.0483, grad_fn=<MeanBackward0>)




```python
deterministic = model.forward(batch, return_scope='item_index', return_type='log_prob', deterministic=True)
random = model.forward(batch, return_scope='item_index', return_type='log_prob', deterministic=False, num_seeds=128)
print(f"{deterministic.shape=:}")
print(f"{random.shape=:}")
# the mean absolute difference between deterministic estimation and random estimation.
torch.mean(torch.abs(deterministic - random.mean(dim=0)))
```

    deterministic.shape=torch.Size([1000])
    random.shape=torch.Size([128, 1000])





    tensor(0.0482, grad_fn=<MeanBackward0>)



## Syntax Sugar: the `predict_proba()` Function
The `predict_proba()` Function mimics the method with the same name in scikit-learn library.

**Note**: to avoid over-flow or under-flow issues, please use the `forward()` function, which provides log-probabilities whenever possible.


```python
@torch.no_grad()
def predict_proba(self, batch: ChoiceDataset) -> torch.Tensor:
    """
    Draw prediction on a given batch of dataset.

    Args:
    batch (ChoiceDataset): the dataset to draw inference on.

    Returns:
    torch.Tensor: the predicted probabilities for each class, the behavior varies by self.pred_item.
    (1: pred_item == True) While predicting items, the return tensor has shape (len(batch), num_items), out[i, j] is the predicted probability for choosing item j AMONG ALL ITEMS IN ITS CATEGORY in observation i. Please note that since probabilities are computed from within-category normalization, hence out.sum(dim=0) can be greater than 1 if there are multiple categories.
    (2: pred_item == False) While predicting external labels for each observations, out[i, 0] is the predicted probability for label == 0 on the i-th observation, out[i, 1] is the predicted probability for label == 1 on the i-th observation. Generally, out[i, 0] + out[i, 1] = 1.0. However, this could be false if under-flowing/over-flowing issue is encountered.
    We highly recommend users to use the forward function to get the log-prob instead.
    """
    pass
```


```python
proba = model.predict_proba(batch)
```

    /Users/tianyudu/miniforge3/envs/ml/lib/python3.9/site-packages/torch/nn/functional.py:1909: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
      warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")



```python
proba
```




    tensor([[0.4757, 0.5243],
            [0.5466, 0.4534],
            [0.5449, 0.4551],
            ...,
            [0.4290, 0.5710],
            [0.5680, 0.4320],
            [0.5732, 0.4268]])



Let's verify that each row of `proba` sum to one:


```python
torch.all(proba.sum(dim=1) == 1.0)
```




    tensor(True)



And the second column of `proba` should be the `prob_positive_class` we calculated above:


```python
torch.all(prob_positive_class == proba[:, 1])
```




    tensor(True)



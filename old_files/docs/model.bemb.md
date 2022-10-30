<!-- markdownlint-disable -->

<a href="../deepchoice/model/bemb.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `model.bemb`






---

<a href="../deepchoice/model/bemb.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VariationalFactorizedGaussian`
A helper class initializes a batch of factorized (i.e., Gaussian distribution with diagional standard covariance matrix) Gaussian distributions. This class is used as the variational family for real-valued latent variables. 

<a href="../deepchoice/model/bemb.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(num_classes: int, dim: int) → None
```



**Args:**
 
 - <b>`num_classes`</b> (int):  the number of Gaussian distributions to create. For example, if we  want the variational distribution of each user's latent to be a 10-dimensional Gaussian,  then num_classes is set to the number of users. The same holds while we are creating  variational distribution for item latent variables. 
 - <b>`dim`</b> (int):  the dimension of each Gaussian distribution. In above example, dim is set to 10. 


---

#### <kbd>property</kbd> device







---

<a href="../deepchoice/model/bemb.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_prob`

```python
log_prob(value: Tensor) → Tensor
```

For each batch B and class C, computes the log probability of value[B, C, :] under the  C-th Gaussian distribution. See the doc string for `batch_factorized_gaussian_log_prob`  for more details. 



**Args:**
 
 - <b>`value`</b> (torch.Tensor):  a tensor with shape (batch_size, num_classes, dim_out). 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  a tensor with shape (batch_size, num_classes). 

---

<a href="../deepchoice/model/bemb.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reparameterize_sample`

```python
reparameterize_sample(num_seeds: int = 1) → Tensor
```

Samples from the multivariate Gaussian distribution using the reparameterization trick. 



**Args:**
 
 - <b>`num_seeds`</b> (int):  number of samples generated. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  a tensor of shape (num_seeds, num_classes, dim), where out[:, C, :] follows  the C-th Gaussian distribution. 


---

<a href="../deepchoice/model/bemb.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LearnableGaussianPrior`




<a href="../deepchoice/model/bemb.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dim_in: int, dim_out: int, std: Union[str, float, Tensor] = 1.0) → None
```

Construct a Gaussian distribution for prior of user/item embeddigns, whose mean and standard deviation depends on user/item observables. NOTE: to avoid exploding number of parameters, learnable parameters in this class are shared  across all items/users. NOTE: we have not supported the standard deviation to be dependent on observables yet. 

For example: p(alpha_ik | H_k, obsItem_i) = Gaussian( mean=H_k*obsItem_i, variance=s2obsPrior ) p(beta_ik | H'_k, obsItem_i) = Gaussian( mean=H'_k*obsItem_i, variance=s2obsPrior ) 



**Args:**
 
 - <b>`dim_in`</b> (int):  the number of input features. 
 - <b>`dim_out`</b> (int):  the dimension of latent features. 
 - <b>`std`</b> (Union[str, float]):  the standard deviation of latent features.  Options are 
 - <b>`0. float`</b>:  a pre-specified constant standard deviation for all dimensions of Gaussian. 1. a tensor with length dim_out with pre-specified constant standard devation. 
 - <b>`2. 'learnable_scalar'`</b>:  a learnable standard deviation shared across all dimensions of Gaussian. 
 - <b>`3. 'learnable_vector'`</b>:  use a separate learnable standard deviation for each dimension of Gaussian. 




---

<a href="../deepchoice/model/bemb.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_prob`

```python
log_prob(x_obs: Tensor, value: Tensor) → Tensor
```

Compute the log likelihood of `value` given observables `x_obs`. 



**Args:**
 
 - <b>`x_obs`</b> (torch.Tensor):  a tensor with shape (num_classes, dim_in) such as item observbales  or user observables, where num_classes is corresponding to the number of items or  number of users. 
 - <b>`value`</b> (torch.Tensor):  a tensor of shape (batch_size, num_classes, dim_out). 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  output shape (batch_size, num_classes) 


---

<a href="../deepchoice/model/bemb.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StandardGaussianPrior`
A helper class for evaluating the log_prob of Monte Carlo samples for latent variables on a N(0, 1) prior. 

<a href="../deepchoice/model/bemb.py#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dim_in: int, dim_out: int) → None
```








---

<a href="../deepchoice/model/bemb.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_prob`

```python
log_prob(x_obs: object, value: Tensor) → Tensor
```

Compute the log-likelihood of `value` under N(0, 1) 



**Args:**
 
 - <b>`x_obs`</b> (object):  x_obs is not used at all, it's here to make args of log_prob consistent  with LearnableGaussianPrior.log_prob(). 
 - <b>`value`</b> (torch.Tensor):  a tensor of shape (batch_size, num_classes, dim_out). 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  output shape (batch_size, num_classes) 


---

<a href="../deepchoice/model/bemb.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BEMB`




<a href="../deepchoice/model/bemb.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    num_users: int,
    num_items: int,
    num_sessions: int,
    obs2prior_dict: Dict[str, bool],
    latent_dim: int,
    latent_dim_price: Optional[int] = None,
    trace_log_q: bool = False,
    category_to_item: Dict[str, List[int]] = None,
    likelihood: str = 'within_category',
    num_user_obs: Optional[int] = None,
    num_item_obs: Optional[int] = None,
    num_session_obs: Optional[int] = None,
    num_price_obs: Optional[int] = None,
    num_taste_obs: Optional[int] = None
) → None
```



**Args:**
 
 - <b>`num_users`</b> (int):  number of users. 
 - <b>`num_items`</b> (int):  number of items. 
 - <b>`num_sessions`</b> (int):  number of sessions. 
 - <b>`latent_dim`</b> (int):  dimension of user and item latents. 
 - <b>`latent_dim_price`</b> (int, optional):  the dimension of latents for the price coefficient. 
 - <b>`trace_log_q`</b> (bool, optional):  whether to trace the derivative of varitional likelihood logQ  with respect to variational parameters in the ELBO while conducting gradient update.  Defaults to False. 
 - <b>`category_to_item`</b> (Dict[str, List[int]], optional):  a dictionary with category id or name  as keys, and category_to_item[C] contains the list of item ids belonging to category C.  If None is provided, all items are assumed to be in the same category.  Defaults to None. 
 - <b>`likelihood`</b> (str, optional):  specifiy the method used for computing likelihood  P(item i | user, session, ...).  Options are 
        - 'all': a softmax across all items. 
        - 'within_category': firstly group items by categories and run separate softmax for each category.  Defaults to 'within_category'. 
 - <b>`obs2prior_user`</b> (bool, optional):  whether user observables enter the prior of user latent or not.  Defaults to False. 
 - <b>`num_user_obs`</b> (Optional[int], optional):  number of user observables, required only if  obs2prior_user is True.  Defaults to None. 
 - <b>`obs2prior_item`</b> (bool, optional):  whether item observables enter the prior of item latent or not.  Defaults to False. 
 - <b>`num_item_obs`</b> (Optional[int], optional):  number of item observables, required only if  obs2prior_item or obs2utility_item is True.  Defaults to None. 
 - <b>`item_intercept`</b> (bool, optional):  whether to add item-specifc intercept (lambda term) to utlity or not.  Defaults to False. 
 - <b>`obs2utility_item`</b> (bool, optional):  whether to allow direct effect from item observables to utility or not.  Defaults to False. 
 - <b>`obs2utility_session`</b> (bool, optional):  whether to allow direct effect from session observables  to utility or not.  Defaults to False. 
 - <b>`num_session_obs`</b> (Optional[int], optional):  number of session observables, required only if  obs2utility_session is True. Defaults to None. 


---

#### <kbd>property</kbd> device





---

#### <kbd>property</kbd> num_params







---

<a href="../deepchoice/model/bemb.py#L600"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `elbo`

```python
elbo(batch, num_seeds: int = 1) → Tensor
```

Computes the current ELBO. 



**Args:**
 
 - <b>`batch`</b> (ChoiceDataset):  a ChoiceDataset containing necessary infromation. 
 - <b>`num_seeds`</b> (int, optional):  the number of Monte Carlo samples from variational distributions  to evaluate the expectation in ELBO.  Defaults to 1. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  a scalar tensor of the ELBO estimated from num_seeds Monte Carlo samples. 

---

<a href="../deepchoice/model/bemb.py#L312"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(batch, return_logit: bool = False) → Tensor
```

Computes the log likelihood of choosing each item in each session. 



**Args:**
 
 - <b>`batch`</b> ([type]):  [description] 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  a tensor of shape (num_sessions, num_items) containing the log likelihood  that each item is chosen in each session. 

---

<a href="../model/bemb/get_within_category_accuracy#L343"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_within_category_accuracy`

```python
get_within_category_accuracy(
    log_p_all_items: Tensor,
    label: LongTensor
) → Dict[str, float]
```

A helper function for computing prediction accuracy (i.e., all non-differential metrics) within category. In particular, thie method calculates the accuracy, precision, recall and F1 score. 



This method has the same functionality as the following peusodcode: for C in categories:  # get sessions in which item in category C was purchased.  T <- (t for t in {0,1,..., len(label)-1} if label[t] is in C)  Y <- label[T] 

 predictions = list()  for t in T:  # get the prediction within category for this session.  y_pred = argmax_{items in C} log prob computed before.  predictions.append(y_pred) 

 accuracy = mean(Y == predictions) 

Similarly, this function computes precision, recall and f1score as well. 



**Args:**
 
 - <b>`log_p_all_items`</b> (torch.Tensor):  shape (num_sessions, num_items) the log probability of  choosing each item in each session. 
 - <b>`label`</b> (torch.LongTensor):  shape (num_sessions,), the IDs of items purchased in each session. 



**Returns:**
 
 - <b>`[Dict[str, float]]`</b>:  A dictionary containing performance metrics. 

---

<a href="../deepchoice/model/bemb.py#L422"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_likelihood`

```python
log_likelihood(batch, sample_dict, return_logit: bool = False) → Tensor
```

Computes the log probability of choosing each item in each session based on current model parameters. This method allows for specifying {user, item}_latent_value for Monte Carlo estimation in ELBO. For actual prediction tasks, use the forward() function, which will use means of varitional distributions for user and item latents. 



**Args:**
 
 - <b>`batch`</b> (ChoiceDataset):  a ChoiceDataset object containing relevant information. 
 - <b>`sample_dict`</b> (Dict[str, torch.Tensor]):  Monte Carlo samples for model coefficients  (i.e., those Greek letters).  sample_dict.keys() should be the same as keys of self.obs2prior_dict, i.e., those  greek letters actually enter the functional form of utility.  The value of sample_dict should be tensors of shape (num_seeds, num_classes, dim)  where num_classes in {num_users, num_items, 1}  and dim in {latent_dim(K), num_item_obs, num_user_obs, 1}. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  a tensor of shape (num_seeds, num_sessions, self.num_items), where  out[x, y, z] is the proabbility of choosing item z in session y conditioned on user  and item latents to be the x-th Monte Carlo sample. 

---

<a href="../deepchoice/model/bemb.py#L573"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_prior`

```python
log_prior(batch, sample_dict)
```





---

<a href="../deepchoice/model/bemb.py#L593"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_variational`

```python
log_variational(sample_dict: Dict[str, Tensor]) → Tensor
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

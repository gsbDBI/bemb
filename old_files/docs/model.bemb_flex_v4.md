<!-- markdownlint-disable -->

<a href="../deepchoice/model/bemb_flex_v4.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `model.bemb_flex_v4`
The Bayesian EMBedding (BEMB) model. (Version 4) Adding customized plug-in. 

A futher attempt to speed things up, split the (1) training step, only calculate utilitis for items in categories bought during training (2) compute all utilities during the inference time. NOTE: release candidate, this version is sufficiently optimized for users. 

**Global Variables**
---------------
- **positive_integer**

---

<a href="../deepchoice/model/bemb_flex_v4.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `parse_utility`

```python
parse_utility(utility_string: str) → List[Dict[str, Union[List[str], NoneType]]]
```

A helper funciton parse utility string into a list of additive terms. 



**Example:**
  utility_string = 'lambda_item + theta_user * alpha_item + gamma_user * beta_item * price_obs'  output = [  {  'coefficient': ['lambda_item'],  'observable': None  },  {  'coefficient': ['theta_user', 'alpha_item'],  'observable': None  },  {  'coefficient': ['gamma_user', 'beta_item'],  'observable': 'price_obs'  }  ] 


---

<a href="../deepchoice/model/bemb_flex_v4.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PositiveInteger`








---

<a href="../deepchoice/model/bemb_flex_v4.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BEMBFlex`




<a href="../deepchoice/model/bemb_flex_v4.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    utility_formula: str,
    obs2prior_dict: Dict[str, bool],
    coef_dim_dict: Dict[str, int],
    num_items: int,
    prior_variance: Union[float, Dict[str, float]] = 1.0,
    num_users: Optional[int] = None,
    num_sessions: Optional[int] = None,
    trace_log_q: bool = False,
    category_to_item: Dict[int, List[int]] = None,
    additive_modules: List[object] = None,
    num_user_obs: Optional[int] = None,
    num_item_obs: Optional[int] = None,
    num_session_obs: Optional[int] = None,
    num_price_obs: Optional[int] = None,
    num_taste_obs: Optional[int] = None
) → None
```



**Args:**
 
 - <b>`utility_formula`</b> (str):  a string representing the utility function U[user, item, session].  See documentation for more details in the documentation for the format of formula. 

**Examples:**
  lambda_item  lambda_item + theta_user * alpha_item + zeta_user * item_obs  lambda_item + theta_user * alpha_item + gamma_user * beta_item * price_obs See the doc-string of parse_utility for an example. 

obs2prior_dict (Dict[str, bool]): a dictionary maps coefficient name (e.g., 'lambda_item') to a boolean indicating if observable (e.g., item_obs) enters the prior of the coefficient. 

coef_dim_dict (Dict[str, int]): a dictionary maps coefficient name (e.g., 'lambda_item') to an integer indicating the dimension of coefficeint. For standalone coefficients like U = lamdba_item, the dim should be 1. For factorized coefficients like U = theta_user * alpha_item, the dim should be the  latent dimension of theta and alpha. For coefficients multiplied with observables like U = zeta_user * item_obs, the dim  should be the number of observables in item_obs. For factorized coefficient muplied with observables like U = gamma_user * beta_item * price_obs,  the dim should be the latent dim multiplied by number of observables in price_obs. 

num_items (int): number of items. 

prior_variance (Union[float, Dict[str, float]]): the variance of prior distribution for coefficients. If a float is provided, all priors will be diagonal matrix with prior_variance along the diagonal. If a dictionary is provided, keys of prior_variance should be coefficient names, and the variance of prior of coef_name would be a diagional matrix with prior_variance[coef_name] along the diagonal. Defaults to 1.0, which means all prior have identity matrix as the covariance matrix. 

num_users (int, optional): number of users, required only if coefficient or observable depending on user is in utitliy. Defaults to None. num_sessions (int, optional): number of sessions, required only if coefficient or observable depending on session is in utility. Defaults to None. 

trace_log_q (bool, optional): whether to trace the derivative of varitional likelihood logQ with respect to variational parameters in the ELBO while conducting gradient update. Defaults to False. 

category_to_item (Dict[str, List[int]], optional): a dictionary with category id or name as keys, and category_to_item[C] contains the list of item ids belonging to category C. If None is provided, all items are assumed to be in the same category. Defaults to None. 

num_{user, item, session, price, taste}_obs (int, optional): number of observables of each type of features, only required if observable enters prior. NOTE: currently we only allow coefficient to depend on either user or item, thus only user and item observables can enter the prior of coefficient. Hence session, price, and taste observables are never required, we include it here for completeness. 


---

#### <kbd>property</kbd> device





---

#### <kbd>property</kbd> num_params







---

<a href="../deepchoice/model/bemb_flex_v4.py#L1021"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../deepchoice/model/bemb_flex_v4.py#L239"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(batch, return_logit: bool = False, all_items: bool = True) → Tensor
```

The combined method of computing utilities and log probability. 



**Args:**
 
 - <b>`batch`</b> ([type]):  [description] 
 - <b>`return_logit`</b> (bool):  return the logit (utility) if set to True, otherwise apply  category-wise log-softmax to compute the log-likelihood before returning. 
 - <b>`all_items`</b> (bool):  only reutrn the logit/log-P of the bought items if set to True,  otherwise return the logit/log-P of all items.  Set all_items to False if only need to compute the log-likelihood for validation  purpose.  Set all_items to True only if you acutally need logits/log-P for all items, such  as for computing inclusive values of categories. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  a tensor of shape (len(batch), num_items) if all_items is True.  a tensor of shape (len(batch),) if all_items is False. 

---

<a href="../model/bemb_flex_v4/get_within_category_accuracy#L284"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_within_category_accuracy`

```python
get_within_category_accuracy(
    log_p_all_items: Tensor,
    label: LongTensor
) → Dict[str, float]
```

A helper function for computing prediction accuracy (i.e., all non-differential metrics) within category. In particular, this method calculates the accuracy, precision, recall and F1 score. 



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

<a href="../deepchoice/model/bemb_flex_v4.py#L591"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_likelihood`

```python
log_likelihood(
    batch,
    sample_dict: Dict[str, Tensor],
    return_logit: bool = False
) → Tensor
```

NOTE: this method is more efficient only computes log-likelihood for bought items. 

Computes the log probability of choosing each item in each session based on current model parameters. This method allows for specifying {user, item}_latent_value for Monte Carlo estimation in ELBO. For actual prediction tasks, use the forward() function, which will use means of varitional distributions for user and item latents. 



**Args:**
 
 - <b>`batch`</b> (ChoiceDataset):  a ChoiceDataset object containing relevant information. 
 - <b>`sample_dict`</b> (Dict[str, torch.Tensor]):  Monte Carlo samples for model coefficients  (i.e., those Greek letters).  sample_dict.keys() should be the same as keys of self.obs2prior_dict, i.e., those  greek letters actually enter the functional form of utility.  The value of sample_dict should be tensors of shape (num_seeds, num_classes, dim)  where num_classes in {num_users, num_items, 1}  and dim in {latent_dim(K), num_item_obs, num_user_obs, 1}. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  a tensor of shape (num_seeds, len(batch)), where  out[x, y] is the proabbility of choosing item batch.item[y] in session y  conditioned on latents to be the x-th Monte Carlo sample. 

---

<a href="../deepchoice/model/bemb_flex_v4.py#L367"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_likelihood_all_items`

```python
log_likelihood_all_items(
    batch,
    sample_dict: Dict[str, Tensor],
    return_logit: bool = False
) → Tensor
```

NOTE: this method computes utilities for all items available, this is relatively slow, for training purpose, use self.log_likelihood() instead. 

Computes the log probability of choosing each item in each session based on current model parameters. This method allows for specifying {user, item}_latent_value for Monte Carlo estimation in ELBO. For actual prediction tasks, use the forward() function, which will use means of varitional distributions for user and item latents. 



**Args:**
 
 - <b>`batch`</b> (ChoiceDataset):  a ChoiceDataset object containing relevant information. 
 - <b>`sample_dict`</b> (Dict[str, torch.Tensor]):  Monte Carlo samples for model coefficients  (i.e., those Greek letters).  sample_dict.keys() should be the same as keys of self.obs2prior_dict, i.e., those  greek letters actually enter the functional form of utility.  The value of sample_dict should be tensors of shape (num_seeds, num_classes, dim)  where num_classes in {num_users, num_items, 1}  and dim in {latent_dim(K), num_item_obs, num_user_obs, 1}. 
 - <b>`return_logit`</b> (bool):  if set to True, return the probability, otherwise return the log-P. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  a tensor of shape (num_seeds, len(batch), self.num_items), where  out[x, y, z] is the proabbility of choosing item z in session y conditioned on  latents to be the x-th Monte Carlo sample. 

---

<a href="../deepchoice/model/bemb_flex_v4.py#L964"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_prior`

```python
log_prior(
    batch,
    sample_dict: Dict[str, Tensor]
) → <built-in method scalar_tensor of type object at 0x7f8ba5699ea0>
```

Calculates the log-likelihood of Monte Carlo samples of Bayesian coefficients under their prior distribution. This method assume coefficients are statistically independnet. 



**Args:**
 
 - <b>`batch`</b> ([type]):  a dataset object contains observables for computing the prior distribution  if obs2prior is True. 
 - <b>`sample_dict`</b> (Dict[str, torch.Tensor]):  a dictionary coefficient names to Monte Carlo  samples. 



**Raises:**
 
 - <b>`ValueError`</b>:  [description] 



**Returns:**
 
 - <b>`torch.scalar_tensor`</b>:  [description] 

---

<a href="../deepchoice/model/bemb_flex_v4.py#L1002"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_variational`

```python
log_variational(sample_dict: Dict[str, Tensor]) → Tensor
```

Calculate the log-likelihood of smaples in sample_dict under the current variational distribution. 



**Args:**
 
 - <b>`sample_dict`</b> (Dict[str, torch.Tensor]):   a dictionary coefficient names to Monte Carlo  samples. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  [description] 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

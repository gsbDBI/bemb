<!-- markdownlint-disable -->

<a href="../deepchoice/model/bemb_flex_v2.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `model.bemb_flex_v2`
The Bayesian EMBedding (BEMB) model. An attempt to speed up things. 


---

<a href="../deepchoice/model/bemb_flex_v2.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `parse_utility`

```python
parse_utility(utility_string: str) → list
```






---

<a href="../deepchoice/model/bemb_flex_v2.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PositiveInteger`








---

<a href="../deepchoice/model/bemb_flex_v2.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BEMBFlex`




<a href="../deepchoice/model/bemb_flex_v2.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    utility_formula: str,
    obs2prior_dict: Dict[str, bool],
    coef_dim_dict: Dict[str, int],
    num_items: int,
    num_users: Optional[int] = None,
    num_sessions: Optional[int] = None,
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
 
 - <b>`utility_formula`</b> (str):  a string representing the utility function U[user, item, session].  See documentation for more details in the documentation for the format of formula. 

**Examples:**
  lambda_item  lambda_item + theta_user * alpha_item + zeta_user * item_obs  lambda_item + theta_user * alpha_item + gamma_user * beta_item * price_obs 

obs2prior_dict (Dict[str, bool]): a dictionary maps coefficient name (e.g., 'lambda_item') to a boolean indicating if observable (e.g., item_obs) enters the prior of the coefficient. 

coef_dim_dict (Dict[str, int]): a dictionary maps coefficient name (e.g., 'lambda_item') to an integer indicating the dimension of coefficeint. For standalone coefficients like U = lamdba_item, the dim should be 1. For factorized coefficients like U = theta_user * alpha_item, the dim should be the  latent dimension of theta and alpha. For coefficients multiplied with observables like U = zeta_user * item_obs, the dim  should be the number of observables in item_obs. For factorized coefficient muplied with observables like U = gamma_user * beta_item * price_obs,  the dim should be the latent dim multiplied by number of observables in price_obs. 

num_items (int): number of items. num_users (int, optional): number of users, required only if coefficient or observable depending on user is in utitliy. Defaults to None. num_sessions (int, optional): number of sessions, required only if coefficient or observable depending on session is in utility. Defaults to None. 

trace_log_q (bool, optional): whether to trace the derivative of varitional likelihood logQ with respect to variational parameters in the ELBO while conducting gradient update. Defaults to False. 

category_to_item (Dict[str, List[int]], optional): a dictionary with category id or name as keys, and category_to_item[C] contains the list of item ids belonging to category C. If None is provided, all items are assumed to be in the same category. Defaults to None. 

likelihood (str, optional): specifiy the method used for computing likelihood P(item i | user, session, ...). Options are 
        - 'all': a softmax across all items. 
        - 'within_category': firstly group items by categories and run separate softmax for each category. Defaults to 'within_category'. 

num_{user, item, session, price, taste}_obs (int, optional): number of observables of each type of features, only required if observable enters prior. NOTE: currently we only allow coefficient to depend on either user or item, thus only user and item observables can enter the prior of coefficient. Hence session, price, and taste observables are never required, we include it here for completeness. 


---

#### <kbd>property</kbd> device





---

#### <kbd>property</kbd> num_params







---

<a href="../deepchoice/model/bemb_flex_v2.py#L626"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../deepchoice/model/bemb_flex_v2.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../model/bemb_flex_v2/get_within_category_accuracy#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../deepchoice/model/bemb_flex_v2.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_likelihood`

```python
log_likelihood(
    batch,
    sample_dict: Dict[str, Tensor],
    return_logit: bool = False
) → Tensor
```

Computes the log probability of choosing each item in each session based on current model parameters. This method allows for specifying {user, item}_latent_value for Monte Carlo estimation in ELBO. For actual prediction tasks, use the forward() function, which will use means of varitional distributions for user and item latents. 



**Args:**
 
 - <b>`batch`</b> (ChoiceDataset):  a ChoiceDataset object containing relevant information. 
 - <b>`sample_dict`</b> (Dict[str, torch.Tensor]):  Monte Carlo samples for model coefficients  (i.e., those Greek letters).  sample_dict.keys() should be the same as keys of self.obs2prior_dict, i.e., those  greek letters actually enter the functional form of utility.  The value of sample_dict should be tensors of shape (num_seeds, num_classes, dim)  where num_classes in {num_users, num_items, 1}  and dim in {latent_dim(K), num_item_obs, num_user_obs, 1}. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  a tensor of shape (num_seeds, num_sessions, self.num_items), where  out[x, y, z] is the proabbility of choosing item z in session y conditioned on user  and item latents to be the x-th Monte Carlo sample. 

---

<a href="../deepchoice/model/bemb_flex_v2.py#L569"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../deepchoice/model/bemb_flex_v2.py#L607"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

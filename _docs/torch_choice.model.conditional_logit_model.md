<!-- markdownlint-disable -->

<a href="../torch_choice/model/conditional_logit_model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `model.conditional_logit_model`
Conditional Logit Model, the generalized version of the `cmclogit' command in Stata. This is the most general implementation of the logit model class. 

Author: Tianyu Du Date: Aug. 8, 2021 



---

<a href="../torch_choice/model/conditional_logit_model.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConditionalLogitModel`
The more generalized version of conditional logit model, the model allows for research specific variable types(groups) and different levels of variations for coefficient. 

The model allows for the following levels for variable variations: NOTE: unless the `-full` flag is specified (which means we want to explicitly model coefficients  for all items), for all variation levels related to item (item specific and user-item specific),  the model force coefficients for the first item to be zero. This design follows standard  econometric practice. 


- constant: constant over all users and items, 


- user: user-specific parameters but constant across all items, 


- item: item-specific parameters but constant across all users, parameters for the first item are  forced to be zero. 
- item-full: item-specific parameters but constant across all users, explicitly model for all items. 


- user-item: parameters that are specific to both user and item, parameter for the first item  for all users are forced to be zero. 
- user-item-full: parameters that are specific to both user and item, explicitly model for all items. 

<a href="../torch_choice/model/conditional_logit_model.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    coef_variation_dict: Dict[str, str],
    num_param_dict: Dict[str, int],
    num_items: Optional[int] = None,
    num_users: Optional[int] = None
) → None
```



**Args:**
 
 - <b>`num_items`</b> (int):  number of items in the dataset. 
 - <b>`num_users`</b> (int):  number of users in the dataset. 
 - <b>`coef_variation_dict`</b> (Dict[str, str]):  variable type to variation level dictionary.  Put None or 'zero' if there is no this kind of variable in the model. 
 - <b>`num_param_dict`</b> (Dict[str, int]):  variable type to number of parameters dictionary,  records number of features in each kind of variable.  Put None if there is no this kind of variable in the model. 


---

#### <kbd>property</kbd> num_params







---

<a href="../torch_choice/model/conditional_logit_model.py#L178"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_hessian`

```python
compute_hessian(x_dict, availability, user_index, y) → Tensor
```

Computes the hessian of negaitve log-likelihood (total cross-entropy loss) with respect to all parameters in this model. 



**Args:**
 
 - <b>`x_dict ,availability, user_index`</b>:  see definitions in self._forward. 
 - <b>`y`</b> (torch.LongTensor):  a tensor with shape (num_trips,) of IDs of items actually purchased. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  a (self.num_params, self.num_params) tensor of the Hessian matrix. 

---

<a href="../torch_choice/model/conditional_logit_model.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_std`

```python
compute_std(x_dict, availability, user_index, y) → Dict[str, Tensor]
```

Computes 



**Args:f**
  See definitions in self.compute_hessian. 



**Returns:**
 
 - <b>`Dict[str, torch.Tensor]`</b>:  a dictoinary whose keys are the same as self.coef_dict.keys() the values are standard errors of coefficients in each coefficient group. 

---

<a href="../torch_choice/model/conditional_logit_model.py#L150"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `flatten_coef_dict`

```python
flatten_coef_dict(
    coef_dict: Dict[str, Union[Tensor, Parameter]]
) → Tuple[Tensor, dict]
```

Flattens the coef_dict into a 1-dimension tensor, used for hessian computation. 

---

<a href="../torch_choice/model/conditional_logit_model.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(
    batch,
    manual_coef_value_dict: Optional[Dict[str, Tensor]] = None
) → Tensor
```

The forward function with explicit arguments, this forward function is for internal usages only, reserachers should use the forward() function insetad. 



**Args:**
  batch: 
 - <b>`manual_coef_value_dict`</b> (Optional[Dict[str, torch.Tensor]], optional):  a dictionary with  keys in {'u', 'i'} etc and tensors as values. If provided, the model will force  coefficient to be the provided values and compute utility conditioned on the provided  coefficient values. This feature is useful when the research wishes to plug in particular  values of coefficients and exmaine the utility values. If not provided, the model will  use the learned coefficient values in self.coef_dict.  Defaults to None. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  a tensor of shape (num_trips, num_items) whose (t, i) entry represents  the utility from item i in trip t for the user involved in that trip. 

---

<a href="../torch_choice/model/conditional_logit_model.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `summary`

```python
summary()
```





---

<a href="../torch_choice/model/conditional_logit_model.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `unwrap_coef_dict`

```python
unwrap_coef_dict(
    param: Tensor,
    type2idx: Dict[str, Tuple[int, int]]
) → Dict[str, Tensor]
```

Rebuild coef_dict from output of self.flatten_coef_dict method. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

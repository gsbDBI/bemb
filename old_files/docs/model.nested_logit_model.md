<!-- markdownlint-disable -->

<a href="../deepchoice/model/nested_logit_model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `model.nested_logit_model`
Implementation of the nested logit model, see page 86 of the book "discrete choice methods with simulation" by Train. for more details. 



---

<a href="../deepchoice/model/nested_logit_model.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NestedLogitModel`




<a href="../deepchoice/model/nested_logit_model.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    category_to_item: Dict[object, List[int]],
    category_coef_variation_dict: Dict[str, str],
    category_num_param_dict: Dict[str, int],
    item_coef_variation_dict: Dict[str, str],
    item_num_param_dict: Dict[str, int],
    num_users: Optional[int] = None,
    shared_lambda: bool = False
) → None
```

Initialization method of the nested logit model. 



**Args:**
 
 - <b>`category_to_item`</b> (Dict[object, List[int]]):  a dictionary maps a category ID to a list  of items IDs of the queried category. 


 - <b>`category_coef_variation_dict`</b> (Dict[str, str]):  a dictionary maps a variable type  (i.e., variable group) to the level of variation for the coefficient of this type  of variables. 
 - <b>`category_num_param_dict`</b> (Dict[str, int]):  a dictoinary maps a variable type name to  the number of parameters in this variable group. 


 - <b>`item_coef_variation_dict`</b> (Dict[str, str]):  the same as category_coef_variation_dict but  for item features. 
 - <b>`item_num_param_dict`</b> (Dict[str, int]):  the same as category_num_param_dict but for item  features. 


 - <b>`num_users`</b> (Optional[int], optional):  number of users to be modelled, this is only  required if any of variable type requires user-specific variations.  Defaults to None. 


 - <b>`shared_lambda`</b> (bool):  a boolean indicating whether to enforce the elasticity lambda, which  is the coefficient for inclusive values, to be constant for all categories.  The lambda enters the category-level selection as the following  Utility of choosing category k = lambda * inclusive value of category k  + linear combination of some other category level features  If set to True, a single lambda will be learned for all categories, otherwise, the  model learns an individual lambda for each category.  Defaults to False. 


---

#### <kbd>property</kbd> num_params







---

<a href="../deepchoice/model/nested_logit_model.py#L246"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_constant`

```python
add_constant(x: Tensor, where: str = 'prepend') → Tensor
```

A helper function used to add constant to feature tensor, x has shape (batch_size, num_classes, num_parameters), returns a tensor of shape (*, num_parameters+1). 

---

<a href="../deepchoice/model/nested_logit_model.py#L235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clamp_lambdas`

```python
clamp_lambdas()
```

Restrict values of lambdas to 0 < lambda <= 1 to guarantee the utility maximization property of the model. This method should be called everytime after optimizer.step(). We add a self_clamp_called_flag to remind researchers if this method is not called. 

---

<a href="../deepchoice/model/nested_logit_model.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(batch)
```





---

<a href="../deepchoice/model/nested_logit_model.py#L218"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_likelihood`

```python
log_likelihood(*args)
```





---

<a href="../deepchoice/model/nested_logit_model.py#L221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `negative_log_likelihood`

```python
negative_log_likelihood(batch, y: LongTensor, is_train: bool = True) → Tensor
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

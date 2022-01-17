<!-- markdownlint-disable -->

<a href="../deepchoice/model/coefficient.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `model.coefficient`
The general class of learnable coefficient/weight/parameter. 



---

<a href="../deepchoice/model/coefficient.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Coefficient`




<a href="../deepchoice/model/coefficient.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    variation: str,
    num_params: int,
    num_items: Optional[int] = None,
    num_users: Optional[int] = None
) → None
```

A generic coefficient object storing trainable parameters. 



**Args:**
 
 - <b>`variation`</b> (str):  the degree of variation of this coefficient. For example, the  coefficient can vary by users or items. 
 - <b>`num_params`</b> (int):  number of parameters. 
 - <b>`num_items`</b> (int):  number of items. 
 - <b>`num_users`</b> (Optional[int], optional):  number of users, this is only necessary if  the coefficient varies by users.  Defaults to None. 




---

<a href="../deepchoice/model/coefficient.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(
    x: Tensor,
    user_index: Optional[Tensor] = None,
    manual_coef_value: Optional[Tensor] = None
) → Tensor
```



**Args:**
 
 - <b>`x`</b> (torch.Tensor):  a tensor of shape (num_sessions, num_items, num_params). 
 - <b>`user_index`</b> (Optional[torch.Tensor], optional):  a tensor of shape (num_sessions,)  contain IDs of the user involved in that session. If set to None, assume the same  user is making all decisions.  Defaults to None. 
 - <b>`manual_coef_value`</b> (Optional[torch.Tensor], optional):  a tensor with the same number of  entries as self.coef. If provided, the forward function uses provided values  as coefficient and return the predicted utility, this feature is useful when  the researcher wishes to manually specify vaues for coefficients and examine prediction  with specified coefficient values. If not provided, forward function is executed  using values from self.coef.  Defaults to None. 



**Returns:**
 
 - <b>`torch.Tensor`</b>:  a tensor of shape (num_sessions, num_items) whose (t, i) entry represents  the utility of purchasing item i in session t. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

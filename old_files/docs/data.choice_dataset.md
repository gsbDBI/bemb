<!-- markdownlint-disable -->

<a href="../deepchoice/data/choice_dataset.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `data.choice_dataset`
The dataset class for consumer choice datasets. Supports for uit linux style naming for variables. 



---

<a href="../deepchoice/data/choice_dataset.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ChoiceDataset`




<a href="../deepchoice/data/choice_dataset.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    label: LongTensor,
    user_index: Optional[LongTensor] = None,
    session_index: Optional[LongTensor] = None,
    item_availability: Optional[BoolTensor] = None,
    **kwargs
) → None
```



**Args:**
 
 - <b>`label`</b> (torch.LongTensor):  a tensor of shape num_purchases (batch_size) indicating the ID  of the item bought. 
 - <b>`user_index`</b> (Optional[torch.LongTensor], optional):  used only if there are multiple users  in the dataset, a tensor of shape num_purchases (batch_size) indicating the ID of the  user who purchased. This tensor is used to select the corresponding user observables and  coefficients tighted to the user (like theta_user) for making prediction for that  purchase.  Defaults to None. 
 - <b>`session_index`</b> (Optional[torch.LongTensor], optional):  used only if there are multiple  sessions in the dataset, a tensor of shape num_purchases (batch_size) indicating the  ID of the session when that purchase occurred. This tensor is used to select the correct  session observables or price observables for making prediction for that purchase.  Defaults to None. 
 - <b>`item_availability`</b> (Optional[torch.BoolTensor], optional):  assume all items are available  if set to None. A tensor of shape (num_sessions, num_items) indicating the availability  of each item in each session.  Defaults to None. 

Other Kwargs (Observables): One can specify the following types of observables, where * in shape denotes any positive  integer. Typically * represents the number of observables. 1. user observables must start with 'user_' and have shape (num_users, *) 2. item observables must start with 'item_' and have shape (num_items, *) 3. session observables must start with 'session_' and have shape (num_sessions, *) 4. taste observables (those vary by user and item) must start with `taste_` and have shape  (num_users, num_items, *). 5. price observables (those vary by session and item) must start with `price_` and have  shape (num_sessions, num_items, *) 


---

#### <kbd>property</kbd> device





---

#### <kbd>property</kbd> num_items





---

#### <kbd>property</kbd> num_sessions





---

#### <kbd>property</kbd> num_users





---

#### <kbd>property</kbd> x_dict

Get the x_dict object for used in model's forward function. 



---

<a href="../deepchoice/data/choice_dataset.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_tensor`

```python
apply_tensor(func)
```





---

<a href="../deepchoice/data/choice_dataset.py#L182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clone`

```python
clone()
```





---

<a href="../deepchoice/data/choice_dataset.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to`

```python
to(device)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

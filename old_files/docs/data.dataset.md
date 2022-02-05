<!-- markdownlint-disable -->

<a href="../deepchoice/data/dataset.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `data.dataset`
Default data structure for a consumer choice problem dataset. 



---

<a href="../deepchoice/data/dataset.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CMDataset`
Choice modelling dataset 

<a href="../deepchoice/data/dataset.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    path: Optional[str] = None,
    X: Optional[Dict[str, Tensor]] = None,
    user_onehot: Optional[LongTensor] = None,
    A: Optional[BoolTensor] = None,
    Y: Optional[LongTensor] = None,
    C: Optional[LongTensor] = None,
    device: str = 'cpu'
) → None
```

Constructs the dataloader, reads dataset from disk directly if `path` is provided. Otherwise, the provided X, user_onehot, etc, tensors will be used. 



**Args:**
 
 - <b>`path`</b> (Optional[str], optional):  the path of dataset. Defaults to None. 


 - <b>`X`</b> (Optional[Dict[str, torch.Tensor]]):  a dictionary with keys from variable types, i.e.,  ['u', 'i', 'ui', 't', 'ut', 'it', 'uit']  and have tensors with shape (num_sessions, num_items, num_params).  num_sessions and num_items are consistent over all keys and values, but num_params  may vary. 
 - <b>`For example, X['u'][session_id, item_id, `</b>: ] contains user-specific features of the user involved in session `session_id`, which will conribute to the utility of item `item_id`. Since the feature is user-specific and does not vary across items, 
 - <b>`X['u'][session_id, item_id_1, `</b>: ] == X['i'][session_id, item_id_2, :]. However, for user-item-specific feature in X['ui'], this equality does not hold. Defaults to None. 


 - <b>`user_onehot`</b> (Optional[torch.LongTensor], optional):  A long tensor with shape  (num_sessions, num_users), in which each row is a one-hot encoding of the ID of the  user who was making purchasing decision in this session.  Defaults to None. 


 - <b>`A`</b> (Optional[torch.BoolTensor], optional):  A boolean tensor with shape  (num_sessions, num_items) indicating the aviliability of each item in each session.  Defaults to None. 


 - <b>`Y`</b> (Optional[torch.LongTensor], optional):  A long tensor with shape (num_sessions,) and  takes values from {0, 1, ..., num_items-1} indicating which item was purchased in  each shopping session.  Defaults to None. 


 - <b>`C`</b> (Optional[torch.LongTensor], optional):  A long tensor with shape (num_sessions,) and  takes values from {0, 1, ..., num_categories-1} indicating which category the item  bought in that session belongs to.  Defaults to None. 


 - <b>`device`</b> (str):  location to store the entire dataset. Defaults to 'cpu'. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

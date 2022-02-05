<!-- markdownlint-disable -->

<a href="../deepchoice/data/data_formatter.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `data.data_formatter`
This script contains utility functions to convert between different formats. Please refer to the documentation for formats supported by this package. 

Author: Tianyu Du Date: July 11, 2021 


---

<a href="../deepchoice/data/data_formatter.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tensors_to_stata`

```python
tensors_to_stata() → DataFrame
```






---

<a href="../deepchoice/data/data_formatter.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `stata_to_tensors`

```python
stata_to_tensors(
    df: DataFrame,
    var_cols_dict: Dict[str, Optional[List[str]]],
    session_id: str = 'id',
    item_id: str = 'mode',
    choice: str = 'choice',
    category_id: Optional[str] = None,
    user_id: Optional[str] = None
) → Tuple[Dict[str, Tensor], LongTensor, BoolTensor, LongTensor, LongTensor]
```

Converts Stata format of data to a dictionary of feature tensors and aviliability. 



**Args:**
 
 - <b>`df`</b> (pd.DataFrame):  the main dataframe in Stata's long format. 
 - <b>`var_cols_dict`</b> (Dict[str, Union[List[str], None]]):  a dictionary with keys from ['u', 'i', 'ui', 't', 'ut', 'it', 'uit']  and has list of column names in df as values.  For example, var_cols_dict['u'] is the list of column names in df that are user-specific variables. 
 - <b>`session_id`</b> (str, optional):  the column in df identifying session/trip ID.  Defaults to 'id'. 
 - <b>`item_id`</b> (str, optional):  the column in df identifying item ID.  Defaults to 'mode'. 
 - <b>`choice`</b> (str, optional):  the column in df identifying the chosen item in each session.  Defaults to 'choice'. 
 - <b>`category_id`</b> (Optional[str], optional):  the column in df identifying which category the item in  each row belongs to.  Defaults to None, set to None if all items belong to the same category. 
 - <b>`user_id`</b> (Optional[str], optional):  the column in df identifying which user was involved in  the session associated with each row.  Defaults to None, set to None if all sessions are done by the same user. 



**Returns:**
 
 - <b>`x_dict`</b> (Dict[str, torch.Tensor]):  dictionary with keys from ['u', 'i', 'ui', 't', 'ut', 'it', 'uit'] and 'aviliability'.  For variation keys like 'u' and 'i', out['u'] has shape (num_trips, num_items, num_params)  as values. 
 - <b>`user_onehot`</b> (torch.LongTensor):  the onehot of users in each session, with shape (num_trips, num_users). 
 - <b>`aviliability`</b> (torch.BoolTensor):  a tensor with 0/1 values and has shape (num_trips, num_items)  indicating the aviliability of each item during each shopping session. 
 - <b>`y`</b> (torch.LongTensor):  a tensor with shape (num_trips) indicating which one among all possible values  in df[item_id] is chosen in that session. 
 - <b>`catetory`</b> (torch.LongTensor):  a tensor with shape (num_trips) indicating the category of the current trip session. 


---

<a href="../deepchoice/data/data_formatter.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `stata_to_X_Y_all`

```python
stata_to_X_Y_all(df: DataFrame) → Tuple[Tensor]
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

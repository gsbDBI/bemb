<!-- markdownlint-disable -->

<a href="../deepchoice/data/dataloader_multitask.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `data.dataloader_multitask`




**Global Variables**
---------------
- **transaction_data**


---

<a href="../deepchoice/data/dataloader_multitask.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Dataset_Train_Multitask`




<a href="../deepchoice/data/dataloader_multitask.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    data_path: str,
    meta_path,
    item_stats_file,
    split: str,
    category=None,
    category_list=None,
    item_list=None,
    item_category=None,
    user_list=None,
    user_onehot=False,
    conditional_model=True,
    users_map=None
)
```








---

<a href="../deepchoice/data/dataloader_multitask.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_X_Y_all`

```python
get_X_Y_all()
```





---

<a href="../deepchoice/data/dataloader_multitask.py#L331"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_X_Y_all_emb`

```python
get_X_Y_all_emb()
```





---

<a href="../deepchoice/data/dataloader_multitask.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_all_data`

```python
get_all_data()
```





---

<a href="../deepchoice/data/dataloader_multitask.py#L177"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `report_stats`

```python
report_stats(df)
```






---

<a href="../deepchoice/data/dataloader_multitask.py#L352"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Dataset_Multitask_NN`




<a href="../deepchoice/data/dataloader_multitask.py#L353"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(X, Y)
```









---

<a href="../deepchoice/data/dataloader_multitask.py#L364"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Dataset_Multitask_NN_emb`




<a href="../deepchoice/data/dataloader_multitask.py#L365"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(X, user_onehot, Y)
```











---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

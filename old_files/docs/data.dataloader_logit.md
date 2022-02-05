<!-- markdownlint-disable -->

<a href="../deepchoice/data/dataloader_logit.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `data.dataloader_logit`
The datalaoder for the logit model. 



---

<a href="../deepchoice/data/dataloader_logit.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LogitDataset`
The original dataloader for the conditional logit model. 

<a href="../deepchoice/data/dataloader_logit.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    data_path: str,
    category_we_care: Optional[str, int] = None,
    split: str = 'train',
    item_list=None
)
```








---

<a href="../deepchoice/data/dataloader_logit.py#L214"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_X_Y_all`

```python
get_X_Y_all() → Tuple[Tensor]
```





---

<a href="../deepchoice/data/dataloader_logit.py#L142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_all_data`

```python
get_all_data() → Tuple[Tensor]
```

Returns a complete set of relevant data. 

---

<a href="../deepchoice/data/dataloader_logit.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `report_stats`

```python
report_stats()
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

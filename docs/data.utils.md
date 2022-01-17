<!-- markdownlint-disable -->

<a href="../deepchoice/data/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `data.utils`





---

<a href="../deepchoice/data/utils.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `pivot3d`

```python
pivot3d(
    df: DataFrame,
    dim0: str,
    dim1: str,
    values: Union[str, List[str]]
) → Tensor
```

Creates a tensor of shape (df[dim0].nunique(), df[dim1].nunique(), len(values)) from the provided data frame. 

Example, if dim0 is the column of session ID, dim1 is the column of alternative names, then  out[t, i, k] is the feature values[k] of item i in session t. The returned tensor  has shape (num_sessions, num_items, num_params), which fits the purpose of conditioanl  logit models. 


---

<a href="../deepchoice/data/utils.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_data_loader`

```python
create_data_loader(
    dataset,
    batch_size: int = -1,
    shuffle: bool = False,
    num_workers: int = 0
)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

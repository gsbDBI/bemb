<!-- markdownlint-disable -->

<a href="../deepchoice/model/nested_bemb.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `model.nested_bemb`
Draft for the BEMB model 



---

<a href="../deepchoice/model/nested_bemb.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NestedBEMB`




<a href="../deepchoice/model/nested_bemb.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    num_users: int,
    num_items: int,
    obs2prior_dict_item: dict,
    obs2prior_dict_category: dict,
    category_to_item: Dict[str, List[int]],
    latent_dim_item: int,
    latent_dim_category: int,
    trace_log_q_item: Optional[bool] = False,
    trace_log_q_category: Optional[bool] = False,
    num_user_obs: Optional[int] = None,
    num_item_obs: Optional[int] = None,
    num_category_obs: Optional[int] = None,
    num_session_obs: Optional[int] = None,
    num_price_obs: Optional[int] = None,
    num_taste_obs: Optional[int] = None,
    shared_lambda: bool = False
)
```








---

<a href="../deepchoice/model/nested_bemb.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `elbo`

```python
elbo(batch_item, batch_category, num_seeds: int = 1) → Tensor
```





---

<a href="../deepchoice/model/nested_bemb.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(batch_item, batch_category)
```





---

<a href="../deepchoice/model/nested_bemb.py#L238"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_within_category_accuracy`

```python
get_within_category_accuracy(*args)
```





---

<a href="../deepchoice/model/nested_bemb.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_likelihood`

```python
log_likelihood(
    batch_item,
    batch_category,
    sample_dict_item,
    sample_dict_category
)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

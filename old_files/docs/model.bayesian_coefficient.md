<!-- markdownlint-disable -->

<a href="../deepchoice/model/bayesian_coefficient.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `model.bayesian_coefficient`






---

<a href="../deepchoice/model/bayesian_coefficient.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BayesianCoefficient`




<a href="../deepchoice/model/bayesian_coefficient.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    variation: str,
    num_classes: int,
    obs2prior: bool,
    num_obs: int = 0,
    dim: int = 1,
    prior_variance: float = 1.0
)
```

The Bayesian coefficient object represents a learnable tensor mu_i in R^k, where i is from a family (e.g., user) so there are num_classes * num_obs learnables. 


---

#### <kbd>property</kbd> device





---

#### <kbd>property</kbd> variational_distribution





---

#### <kbd>property</kbd> variational_mean







---

<a href="../deepchoice/model/bayesian_coefficient.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_prior`

```python
log_prior(sample: Tensor, x_obs: Optional[Tensor] = None)
```





---

<a href="../deepchoice/model/bayesian_coefficient.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_variational`

```python
log_variational(sample=None)
```





---

<a href="../deepchoice/model/bayesian_coefficient.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reparameterize_sample`

```python
reparameterize_sample(num_seeds: int = 1)
```





---

<a href="../deepchoice/model/bayesian_coefficient.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_variational_mean_fixed`

```python
update_variational_mean_fixed(new_value: Tensor)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

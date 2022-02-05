<!-- markdownlint-disable -->

<a href="../deepchoice/model/bayesian_linear.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `model.bayesian_linear`
Bayesian tensor object. Objective: this is a generalization of the Bayesian Coefficient object. The Bayesian Tensor is designed to be hierarchical, so it's more than a single tensor, it's a module. TODO: might change to Bayesian Layer or other name. 

For the current iteration, we assume each entry of the weight matrix follows independent normal distributions. TODO: might generalize this setting in the future. TODO: generalize this setting to arbitrary shape tensors. 



---

<a href="../deepchoice/model/bayesian_linear.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BayesianLinear`




<a href="../deepchoice/model/bayesian_linear.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    in_features: int,
    out_features: int,
    bias: bool = True,
    obs2prior: bool = False,
    num_obs: Optional[int] = None,
    variational_weight_mean_fixed: Optional[Tensor] = None,
    device=None,
    dtype=None
)
```

Linear layer where weight and bias are modelled as distributions.  




---

#### <kbd>property</kbd> bias_distribution





---

#### <kbd>property</kbd> device





---

#### <kbd>property</kbd> variational_weight_mean





---

#### <kbd>property</kbd> weight_distribution

the weight variational distribution. 



---

<a href="../deepchoice/model/bayesian_linear.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(
    x,
    num_seeds: int = 1,
    deterministic: bool = False,
    mode: str = 'multiply'
)
```

Forward with weight sampling. Forward does out = XW + b, for forward() method behaves like the embedding layer in PyTorch, use the lookup() method. If deterministic, use the mean. mode in ['multiply', 'lookup'] 

---

<a href="../deepchoice/model/bayesian_linear.py#L203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `gen_sample`

```python
gen_sample()
```

Sample parameters and store locally. 

---

<a href="../deepchoice/model/bayesian_linear.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_prior`

```python
log_prior(
    W_sample: Tensor,
    b_sample: Optional[Tensor] = None,
    H_sample: Optional[Tensor] = None,
    x_obs: Optional[Tensor] = None
)
```

Evaluate the likelihood of the provided samples of parameter under the current prior distribution. 

---

<a href="../deepchoice/model/bayesian_linear.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `log_variational`

```python
log_variational(W_sample: Tensor, b_sample: Optional[Tensor] = None)
```

Evaluate the likelihood of the provided samples of parameter under the current variational distribution. 

---

<a href="../deepchoice/model/bayesian_linear.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `rsample`

```python
rsample(num_seeds: int = 1)
```

sample all parameters using re-parameterization trick. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

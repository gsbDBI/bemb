<!-- markdownlint-disable -->

<a href="../deepchoice/model/bemb_lightning.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `model.bemb_lightning`






---

<a href="../deepchoice/model/bemb_lightning.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LitBEMB`




<a href="../deepchoice/model/bemb_lightning.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(learning_rate: float = 0.3, num_seeds: int = 1, **kwargs)
```






---

#### <kbd>property</kbd> automatic_optimization

If set to ``False`` you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()``. 

---

#### <kbd>property</kbd> current_epoch

The current epoch in the Trainer. If no Trainer is attached, this propery is 0. 

---

#### <kbd>property</kbd> datamodule





---

#### <kbd>property</kbd> device





---

#### <kbd>property</kbd> dtype





---

#### <kbd>property</kbd> example_input_array

The example input array is a specification of what the module can consume in the :meth:`forward` method. The return type is interpreted as follows: 


-   Single tensor: It is assumed the model takes a single argument, i.e.,  ``model.forward(model.example_input_array)`` 
-   Tuple: The input array should be interpreted as a sequence of positional arguments, i.e.,  ``model.forward(*model.example_input_array)`` 
-   Dict: The input array represents named keyword arguments, i.e.,  ``model.forward(**model.example_input_array)`` 

---

#### <kbd>property</kbd> global_rank

The index of the current process across all nodes and devices. 

---

#### <kbd>property</kbd> global_step

Total training batches seen across all epochs. If no Trainer is attached, this propery is 0. 

---

#### <kbd>property</kbd> hparams





---

#### <kbd>property</kbd> hparams_initial





---

#### <kbd>property</kbd> loaded_optimizer_states_dict





---

#### <kbd>property</kbd> local_rank

The index of the current process within a single node. 

---

#### <kbd>property</kbd> logger

Reference to the logger object in the Trainer. 

---

#### <kbd>property</kbd> model_size

The model's size in megabytes. The computation includes everything in the :meth:`~torch.nn.Module.state_dict`, i.e., by default the parameteters and buffers. 

---

#### <kbd>property</kbd> on_gpu

Returns ``True`` if this model is currently located on a GPU. Useful to set flags around the LightningModule for different CPU vs GPU behavior. 

---

#### <kbd>property</kbd> truncated_bptt_steps

Enables `Truncated Backpropagation Through Time` in the Trainer when set to a positive integer. It represents the number of times :meth:`training_step` gets called before backpropagation. If this is > 0, the :meth:`training_step` receives an additional argument ``hiddens`` and is expected to return a hidden state. 



---

<a href="../deepchoice/model/bemb_lightning.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `configure_optimizers`

```python
configure_optimizers()
```





---

<a href="../deepchoice/model/bemb_lightning.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluation_step`

```python
evaluation_step(batch, batch_idx, name)
```





---

<a href="../deepchoice/model/bemb_lightning.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `test_step`

```python
test_step(batch, batch_idx)
```





---

<a href="../deepchoice/model/bemb_lightning.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `training_step`

```python
training_step(batch, batch_idx)
```





---

<a href="../deepchoice/model/bemb_lightning.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `validation_step`

```python
validation_step(batch, batch_idx)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

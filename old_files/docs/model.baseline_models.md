<!-- markdownlint-disable -->

<a href="../deepchoice/model/baseline_models.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `model.baseline_models`
Collection of baseline models. 



---

<a href="../deepchoice/model/baseline_models.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseMNLNoMask`
Simplest multinomial logit model. Considers all possible output classes, regardless of availability. Features used are user-specific and the coefficients are item-specific. 

<a href="../deepchoice/model/baseline_models.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(feature_dim, class_dim)
```








---

<a href="../deepchoice/model/baseline_models.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```






---

<a href="../deepchoice/model/baseline_models.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseMNL`
Simplest multinomial logit model with masking. In other words, only allows predictions for those classes for which a particular session has availability. Features used are user-specific and the coefficients are item-specific. 

<a href="../deepchoice/model/baseline_models.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(feature_dim, class_dim)
```








---

<a href="../deepchoice/model/baseline_models.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x, mask)
```






---

<a href="../deepchoice/model/baseline_models.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DeepMNL`
A "deep" multinomial logit model. This one has num_layers number of hidden layers between input and output to allow for a deeper representation of the response surface. Besides the hidden layers, this is otherwise identical to BaseMNL. 

<a href="../deepchoice/model/baseline_models.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(feature_dim, class_dim, size_layer, num_layers)
```








---

<a href="../deepchoice/model/baseline_models.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x, mask)
```






---

<a href="../deepchoice/model/baseline_models.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `InterceptMNL`
This is an intercept MNL model. That means the input features are one-hots for persona_1 and the outputs are probabilities of reaching persona_2. Since this could easily become intractible with millions of parameters to train, we mask out most of them using a persona_1 to persona_2 availability list. 

__init__ arguments: coef_mask -- a tensor of dimension feature_dim (number of persona_1) by class_dim (number of persona_2). It is an availability list at the persona_1 to persona_2 transition level. 

forward arguments: mask -- a tensor of dimension x.size(1) by class_dim. It is an availability list at the session to persona_2 transition level. 

<a href="../deepchoice/model/baseline_models.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(feature_dim, class_dim, coef_mask)
```








---

<a href="../deepchoice/model/baseline_models.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x, mask)
```






---

<a href="../deepchoice/model/baseline_models.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `InertiaMNL`
A baseline multinomial logit model with inertia. 

<a href="../deepchoice/model/baseline_models.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="../deepchoice/model/baseline_models.py#L123"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x_inertia, mask)
```






---

<a href="../deepchoice/model/baseline_models.py#L130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConditionalLogit`
A baseline conditional logit model. Here we use features of _choices_ (x_choice), the conditional part, and it uses the same coefficient for all choices within the same feature. We also use features of user (x_user), the multinomial logit part, and they have item-specific coefficients. This model also allows for varying choice sets through mask. 

<a href="../deepchoice/model/baseline_models.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(user_feature_dim, choice_feature_dim, class_dim)
```








---

<a href="../deepchoice/model/baseline_models.py#L150"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x_user, x_choice, mask)
```






---

<a href="../deepchoice/model/baseline_models.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `InterceptConditionalLogit`
An intercept conditional logit model. This includes 3 components: 1) an MNL part using persona_1 one-hots with a coefficient mask to zero out coefficients not in the availability list, 2) an MNL part for user features, 3) a conditional logit part with persona_2 and price features. 

__init__ arguments: user_intercept_dim -- number of persona_1. The categorical variable persona_1 is represented as one-hots. user_feature_dim -- number of persona_1 level features choice_feature_dim -- number of persona_2 level features class_dim -- number of persona_2 coef_mask -- a tensor of dimension feature_dim (number of persona_1) by class_dim (number of persona_2). It is an availability list at the persona_1 to persona_2 transition level. 

forward arguments: x_intercept -- tensor whose size is num_sessions x user_intercept_dim. It is a one-hot expansion of the cateogorical variable, persona_1. x_user -- tensor of size num_sessions x user_feature_dim. These are the persona_1 features x_choice -- tensor of size num_sessions x choice_feature_dim x class_dim. These are the "widened" persona_2 features. mask -- tensor of size num_sessions x class_dim. This represents the session to persona_2 availability list. 

<a href="../deepchoice/model/baseline_models.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    user_intercept_dim,
    user_feature_dim,
    choice_feature_dim,
    class_dim,
    coef_mask
)
```








---

<a href="../deepchoice/model/baseline_models.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x_intercept, x_user, x_choice, mask)
```






---

<a href="../deepchoice/model/baseline_models.py#L219"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `InertiaConditionalLogit`
A conditional logit model with inertia. 

<a href="../deepchoice/model/baseline_models.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(user_feature_dim, choice_feature_dim, class_dim)
```








---

<a href="../deepchoice/model/baseline_models.py#L236"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x_inertia, x_user, x_choice, mask)
```






---

<a href="../deepchoice/model/baseline_models.py#L249"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LeanConditionalLogit`
Similar to ConditionalLogit but reduces memory consumption by separating clogit features that depend only on persona_2 (and therefore don't need to be copied for every session) and those that depend both on persona_2 and session (which will be treated as before). 

__init__ arguments: user_feature_dim -- number of persona_1 level features price_dim -- number of price features, i.e. those that vary by both session and persona_2 choice_feature_dim -- number of persona_2 level features class_dim -- number of persona_2 

forward arguments: x_user -- tensor of size num_sessions x user_feature_dim. These are the persona_1 features x_price -- tensor of size num_sessions x price_dim x class_dim. These are the "widened" price features. x_choice -- tensor of size choice_feature_dim x class_dim. These are the persona_2 features independent of sessions. mask -- tensor of size num_sessions x class_dim. This represents the session to persona_2 availability list. 

<a href="../deepchoice/model/baseline_models.py#L269"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(user_feature_dim, price_dim, choice_feature_dim, class_dim)
```








---

<a href="../deepchoice/model/baseline_models.py#L283"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x_user, x_price, x_choice, mask)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

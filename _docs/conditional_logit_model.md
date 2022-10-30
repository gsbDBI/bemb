<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Random Utility Model (RUM) Part I: Conditional Logit Model
The first part of random utility model (RUM) documentation covers the conditional logit model and provides an example usage of the conditional logit model.
This documentation assumes the reader has already gone through the [data management tutorial](./data_management.md).

This tutorial is adopted from the [Random utility model and the multinomial logit model](https://cran.r-project.org/web/packages/mlogit/vignettes/c3.rum.html) in th documentation of `mlogit` package in R.
Please refer to this tutorial for a complete treatment on the mathematical theory behind the conditional logit model.

Please note that the dataset involved in this example is relatively small (2,779 choice records), so we don't expect the performance to be faster than the R implementation. We provide this tutorial mainly to check the correctness of our prediction. The fully potential of PyTorch is better exploited on much larger dataset.

We have provided a Jupyter notebook version of this tutorial [here](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/conditional_logit_model_mode_canada.ipynb) as well.

## Load Required Dependencies
We first load required dependencies for this tutorial.
```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch_choice.data import ChoiceDataset, utils
from torch_choice.model import ConditionalLogitModel

from torch_choice.utils.run_helper import run
```

Now we check if there's any CUDA-compatible hardware installed.
```python
if torch.cuda.is_available():
    print(f'CUDA device used: {torch.cuda.get_device_name()}')
    device = 'cuda'
else:
    device = 'cpu'
```
    CUDA device used: NVIDIA GeForce RTX 3090


## Load the Travel Mode Dataset
This tutorial uses the `ModeCanada` dataset for people's choice on travelling methods.

```python
df = pd.read_csv('./public_datasets/ModeCanada.csv', index_col=0)
df = df.query('noalt == 4').reset_index(drop=True)
df.sort_values(by='case', inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>case</th>
      <th>alt</th>
      <th>choice</th>
      <th>dist</th>
      <th>cost</th>
      <th>ivt</th>
      <th>ovt</th>
      <th>freq</th>
      <th>income</th>
      <th>urban</th>
      <th>noalt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>109</td>
      <td>train</td>
      <td>0</td>
      <td>377</td>
      <td>58.25</td>
      <td>215</td>
      <td>74</td>
      <td>4</td>
      <td>45</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>109</td>
      <td>air</td>
      <td>1</td>
      <td>377</td>
      <td>142.80</td>
      <td>56</td>
      <td>85</td>
      <td>9</td>
      <td>45</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>109</td>
      <td>bus</td>
      <td>0</td>
      <td>377</td>
      <td>27.52</td>
      <td>301</td>
      <td>63</td>
      <td>8</td>
      <td>45</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>109</td>
      <td>car</td>
      <td>0</td>
      <td>377</td>
      <td>71.63</td>
      <td>262</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>110</td>
      <td>train</td>
      <td>0</td>
      <td>377</td>
      <td>58.25</td>
      <td>215</td>
      <td>74</td>
      <td>4</td>
      <td>70</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (11116, 11)




```python
label = df[df['choice'] == 1].sort_values(by='case')['alt'].reset_index(drop=True)
print(f"{label=:}")
```

    label=0       air
    1       air
    2       air
    3       air
    4       air
           ...
    2774    car
    2775    car
    2776    car
    2777    car
    2778    car
    Name: alt, Length: 2779, dtype: object



```python
item_names = ['air', 'bus', 'car', 'train']
num_items = 4
encoder = dict(zip(item_names, range(num_items)))
print(f"{encoder=:}")
label = label.map(lambda x: encoder[x])
label = torch.LongTensor(label)
print(f"{label=:}")
```

    encoder={'air': 0, 'bus': 1, 'car': 2, 'train': 3}
    label=tensor([0, 0, 0,  ..., 2, 2, 2])



```python
price_cost_freq_ovt = utils.pivot3d(df, dim0='case', dim1='alt',
                                    values=['cost', 'freq', 'ovt'])
session_income = df.groupby('case')['income'].first()
session_income = torch.Tensor(session_income.values).view(-1, 1)
price_ivt = utils.pivot3d(df, dim0='case', dim1='alt', values='ivt')
```


```python
dataset= ChoiceDataset(label=label,
                       price_cost_freq_ovt=price_cost_freq_ovt,
                       session_income=session_income,
                       price_ivt=price_ivt
                       ).to(device)
```


```python
dataset
```




    ChoiceDataset(label=[2779], user_index=[], session_index=[2779], item_availability=[], observable_prefix=[5], price_cost_freq_ovt=[2779, 4, 3], session_income=[2779, 1], price_ivt=[2779, 4, 1], device=cuda:0)


The `ChoiceDataset` constructed contains 2779 choice records. Since the original dataset did not reveal the identity of each decision maker, we consider all 2779 choices were made by a single user but in 2779 different sessions to handle variations.

In this case, the `cost`, `freq` and `ovt` are observables depending on both sessions and items, we created a `price_cost_freq_ovt` tensor with shape `(num_sessions, num_items, 3) = (2779, 4, 3)` to contain these variables.
In contrast, the `income` information depends only on session but not on items, hence we create the `session_income` tensor to store it.

Because we wish to fit item-specific coefficients for the `ivt` variable, which varies by both sessions and items as well, we create another `price_ivt` tensor in addition to the `price_cost_freq_ovt` tensor.

## Create the Model
We aim to estimate the following model formulation:
$$
U_{uit} = \beta^0_i + \beta^{1\top} X^{price: (cost, freq, ovt)}_{it} + \beta^2_i X^{session:income}_t + \beta^3_i X_{it}^{price:ivt} + \epsilon_{uit}
$$

We now initialize the `ConditionalLogitModel` to predict choices from the dataset. Please see the documentation [here](./torch_choice.model.conditional_logit_model.md) for a complete description of the `ConditionalLogitModel` class.

The `ConditionalLogitModel` constructor requires four components:
1. `coef_variation_dict` is a dictionary with variable names (defined above while constructing the dataset) as keys and values from `{constant, user, item, item-full}`. For instance, since we wish to have constant coefficients for `cost`, `freq` and `ovt` observables, and these three observables are stored in the `price_cost_freq_ovt` tensor of the choice dataset, we set `coef_variation_dict['price_cost_freq_ovt'] = 'constant'`.
  The models allows for the option of enforcing coefficient for one item to be zero, the variation of $$\beta^3$$ is specified as `item-full` which indicates 4 values of $$\beta^3$$ is learned (one for each item). In contrast, $$\beta^0, \beta^2$$ are specified to have variation `item` instead of `item-full`. In this case, the $$\beta$$ correspond to the first item (i.e., the baseline item, which is encoded as 0 in the label tensor) is force to be zero.
   The researcher needs to declare `intercept` (as shown below) manually for the model to fit an intercept as well, otherwise the model assumes zero intercept term.
2. `num_param_dict` is a dictionary with keys exactly the same as the `coef_variation_dict`. Each of dictionary values tells the dimension of the corresponding observables (hence the dimension of the coefficient). For example, the `price_cost_freq_ovt` consists of three observables and we set the corresponding to three.
   Even the model can infer `num_param_dict['intercept'] = 1`, but we recommend the research to include it for completeness.
3. `num_items` informs the model how many alternatives users are choosing from.
4. `num_users` is an optional integer informing the model how many users there are in the dataset. However, in this example we implicitly assume there is only one user making all the decisions and we do not have any `user_obs` involved, hence `num_users` argument is not supplied.

```python
model = ConditionalLogitModel(coef_variation_dict={'price_cost_freq_ovt': 'constant',
                                                   'session_income': 'item',
                                                   'price_ivt': 'item-full',
                                                   'intercept': 'item'},
                              num_param_dict={'price_cost_freq_ovt': 3,
                                              'session_income': 1,
                                              'price_ivt': 1,
                                              'intercept': 1},
                              num_items=4)

model = model.to(device)
```
## Fit the Model
We provide an easy-to-use model runner for both `ConditionalLogitModel` and `NestedLogitModel` instances.

```python
run(model, dataset, num_epochs=10000)
```

    ==================== received model ====================
    ConditionalLogitModel(
      (coef_dict): ModuleDict(
        (price_cost_freq_ovt): Coefficient(variation=constant, num_items=4, num_users=None, num_params=3, 3 trainable parameters in total).
        (session_income): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total).
        (price_ivt): Coefficient(variation=item-full, num_items=4, num_users=None, num_params=1, 4 trainable parameters in total).
        (intercept): Coefficient(variation=item, num_items=4, num_users=None, num_params=1, 3 trainable parameters in total).
      )
    )
    Conditional logistic discrete choice model, expects input features:

    X[price_cost_freq_ovt] with 3 parameters, with constant level variation.
    X[session_income] with 1 parameters, with item level variation.
    X[price_ivt] with 1 parameters, with item-full level variation.
    X[intercept] with 1 parameters, with item level variation.
    ==================== received dataset ====================
    ChoiceDataset(label=[2779], user_index=[], session_index=[2779], item_availability=[], observable_prefix=[5], price_cost_freq_ovt=[2779, 4, 3], session_income=[2779, 1], price_ivt=[2779, 4, 1], device=cuda:0)
    ==================== training the model ====================
    Epoch 1000: Mean Log-likelihood=-1879.8817138671875
    Epoch 2000: Mean Log-likelihood=-1877.4864501953125
    Epoch 3000: Mean Log-likelihood=-1884.6649169921875
    Epoch 4000: Mean Log-likelihood=-1876.64453125
    Epoch 5000: Mean Log-likelihood=-1876.778564453125
    Epoch 6000: Mean Log-likelihood=-1876.9173583984375
    Epoch 7000: Mean Log-likelihood=-1878.3720703125
    Epoch 8000: Mean Log-likelihood=-1874.887939453125
    Epoch 9000: Mean Log-likelihood=-1878.2408447265625
    Epoch 10000: Mean Log-likelihood=-1874.84912109375
    ==================== model results ====================
    Training Epochs: 10000

    Learning Rate: 0.01

    Batch Size: 2779 out of 2779 observations in total

    Final Log-likelihood: -1874.84912109375

    Coefficients:

    | Coefficient           |   Estimation |   Std. Err. |
    |:----------------------|-------------:|------------:|
    | price_cost_freq_ovt_0 |  -0.0349012  |  0.00714914 |
    | price_cost_freq_ovt_1 |   0.0932566  |  0.0051167  |
    | price_cost_freq_ovt_2 |  -0.0431938  |  0.00325467 |
    | session_income_0      |  -0.0885135  |  0.0184003  |
    | session_income_1      |  -0.0276497  |  0.00385565 |
    | session_income_2      |  -0.037754   |  0.00410114 |
    | price_ivt_0           |   0.0596341  |  0.0101276  |
    | price_ivt_1           |  -0.00678102 |  0.00443506 |
    | price_ivt_2           |  -0.00613108 |  0.001907   |
    | price_ivt_3           |  -0.00150583 |  0.00119624 |
    | intercept_0           |   0.459606   |  1.28338    |
    | intercept_1           |   1.6092     |  0.709882   |
    | intercept_2           |   3.07506    |  0.625779   |



## R Output
The following is the R-output from the `mlogit` implementation, the estimation, standard error, and log-likelihood from our `torch_choice` implementation is the same as the result from `mlogit` implementation.
```r
install.packages("mlogit")
library("mlogit")
data("ModeCanada", package = "mlogit")
MC <- dfidx(ModeCanada, subset = noalt == 4)
ml.MC1 <- mlogit(choice ~ cost + freq + ovt | income | ivt, MC, reflevel='air')

summary(ml.MC1)
```
```
Call:
mlogit(formula = choice ~ cost + freq + ovt | income | ivt, data = MC,
    reflevel = "air", method = "nr")

Frequencies of alternatives:choice
      air     train       bus       car
0.3738755 0.1666067 0.0035984 0.4559194

nr method
9 iterations, 0h:0m:0s
g'(-H)^-1g = 0.00014
successive function values within tolerance limits

Coefficients :
                    Estimate Std. Error  z-value  Pr(>|z|)
(Intercept):train  3.2741952  0.6244152   5.2436 1.575e-07 ***
(Intercept):bus    0.6983381  1.2802466   0.5455 0.5854292
(Intercept):car    1.8441129  0.7085089   2.6028 0.0092464 **
cost              -0.0333389  0.0070955  -4.6986 2.620e-06 ***
freq               0.0925297  0.0050976  18.1517 < 2.2e-16 ***
ovt               -0.0430036  0.0032247 -13.3356 < 2.2e-16 ***
income:train      -0.0381466  0.0040831  -9.3426 < 2.2e-16 ***
income:bus        -0.0890867  0.0183471  -4.8556 1.200e-06 ***
income:car        -0.0279930  0.0038726  -7.2286 4.881e-13 ***
ivt:air            0.0595097  0.0100727   5.9080 3.463e-09 ***
ivt:train         -0.0014504  0.0011875  -1.2214 0.2219430
ivt:bus           -0.0067835  0.0044334  -1.5301 0.1259938
ivt:car           -0.0064603  0.0018985  -3.4029 0.0006668 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Log-Likelihood: -1874.3
McFadden R^2:  0.35443
Likelihood ratio test : chisq = 2058.1 (p.value = < 2.22e-16)
```
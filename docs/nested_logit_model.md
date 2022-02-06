<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Random Utility Model (RUM) Part II: Nested Logit Model
The package implements the nested logit model as well, which allows researchers to model choices as a two-stage process: the user first picks a category of purchase and then picks the item from the chosen category that generates the most utility.

This documentation assumes the reader has already gone through the [data management tutorial](./data_management.md).

We also provide a Jupyter notebook version of examples below [here](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/nested_logit_model_house_cooling.ipynb).
## Nested Logit Model
The following code block provides an example initialization of the `NestedLogitModel` (please refer to examples below for details).
```python
model = NestedLogitModel(category_to_item=category_to_item,
                         category_coef_variation_dict={},
                         category_num_param_dict={},
                         item_coef_variation_dict={'price_obs': 'constant'},
                         item_num_param_dict={'price_obs': 7},
                         shared_lambda=True)
```

The nested logit model decompose the utility of choosing item $$i$$ into the (1) item-specific values and (2) category specify values.  For simplicity, suppose item $i$  belongs to category $$k \in \{1, \dots, K\}$$: $$i \in B_k$$.
$$
U_{uit} = W_{ukt} + Y_{uit}
$$
Where both $$W$$ and $$Y$$ are estimated using linear models from as in the conditional logit model.

The log-likelihood for user $$u$$ to choose item $$i$$ at time/session $$t$$ decomposes into the item-level likelihood and category-level likelihood.
$$
\log P(i \mid u, t) = \log P(i \mid u, t, B_k) + \log P(k \mid u, t) \\
= \log \left(\frac{\exp(Y_{uit}/\lambda_k)}{\sum_{j \in B_k} \exp(Y_{ujt}/\lambda_k)}\right) + \log \left( \frac{\exp(W_{ukt} + \lambda_k I_{ukt})}{\sum_{\ell=1}^K \exp(W_{u\ell t} + \lambda_\ell I_{u\ell t})}\right)
$$
The **inclusive value** of category $$k$$, $$I_{ukt}$$ is defined as $$\log \sum_{j \in B_k} \exp(Y_{ujt}/\lambda_k)$$, which is the *expected utility from choosing the best alternative from category $k$*.

The `category_to_item` keyword defines a dictionary of the mapping $$k \mapsto B_k$$, where keys of `category_to_item`  are integer $$k$$'s and  `category_to_item[k]`  is a list consisting of IDs of items in $$B_k$$.

The `{category, item}_coef_variation_dict` provides specification to $$W_{ukt}$$ and $$Y_{uit}$$ respectively, `torch_choice` allows for empty category level models by providing an empty dictionary (in this case, $$W_{ukt} = \epsilon_{ukt}$$) since the inclusive value term $$\lambda_k I_{ukt}$$ will be used to model the choice over categories. However, by specifying an empty second stage model ($$Y_{uit} = \epsilon_{uit}$$), the nested logit model reduces to a conditional logit model of choices over categories. Hence, one should never use the `NestedLogitModel` class with an empty item-level model.

Similar to the conditional logit model, `{category, item}_num_param_dict` specify the dimension (number of observables to be multiplied with the coefficient) of coefficients. The above code initializes a simple model built upon item-time-specific observables $$X_{it} \in \mathbb{R}^7$$,
$$
Y_{uit} = \beta^\top X_{it} + \epsilon_{uit} \\
W_{ukt} = \epsilon_{ukt}
$$
The research may wish to enfoce the *elasiticity* $$\lambda_k$$ to be constant across categories, setting `shared_lambda=True` enforces $$\lambda_k = \lambda\ \forall k \in [K]$$.


# Examples on Nested Logit Model
Author: Tianyu Du
Examples here are modified from [Exercise 2: Nested logit model by Kenneth Train and Yves Croissant](https://cran.r-project.org/web/packages/mlogit/vignettes/e2nlogit.html)
The data set HC from mlogit contains data in R format on the choice of heating and central cooling system for 250 single-family, newly built houses in California.

The alternatives are:

- Gas central heat with cooling gcc,
- Electric central resistence heat with cooling ecc,
- Electric room resistence heat with cooling erc,
- Electric heat pump, which provides cooling also hpc,
- Gas central heat without cooling gc,
- Electric central resistence heat without cooling ec,
- Electric room resistence heat without cooling er.
- Heat pumps necessarily provide both heating and cooling such that heat pump without cooling is not an alternative.

The variables are:

- depvar gives the name of the chosen alternative,
- ich.alt are the installation cost for the heating portion of the system,
- icca is the installation cost for cooling
- och.alt are the operating cost for the heating portion of the system
- occa is the operating cost for cooling
- income is the annual income of the household

Note that the full installation cost of alternative gcc is ich.gcc+icca, and similarly for the operating cost and for the other alternatives with cooling.

## Load Essential Packages.

```python
import pandas as pd
import torch

from torch_choice.data import ChoiceDataset, JointDataset, utils
from torch_choice.model.nested_logit_model import NestedLogitModel
from torch_choice.utils.run_helper import run

if torch.cuda.is_available():
    print(f'CUDA device used: {torch.cuda.get_device_name()}')
    device = 'cuda'
else:
    device = 'cpu'
```

    CUDA device used: NVIDIA GeForce RTX 3090

## Load Datasets

```python
df = pd.read_csv('./public_datasets/HC.csv', index_col=0)
df = df.reset_index(drop=True)
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
      <th>depvar</th>
      <th>icca</th>
      <th>occa</th>
      <th>income</th>
      <th>ich</th>
      <th>och</th>
      <th>idx.id1</th>
      <th>idx.id2</th>
      <th>inc.room</th>
      <th>inc.cooling</th>
      <th>int.cooling</th>
      <th>cooling.modes</th>
      <th>room.modes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>20</td>
      <td>24.50</td>
      <td>4.09</td>
      <td>1</td>
      <td>ec</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>27.28</td>
      <td>2.95</td>
      <td>20</td>
      <td>7.86</td>
      <td>4.09</td>
      <td>1</td>
      <td>ecc</td>
      <td>0</td>
      <td>20</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>20</td>
      <td>7.37</td>
      <td>3.85</td>
      <td>1</td>
      <td>er</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>27.28</td>
      <td>2.95</td>
      <td>20</td>
      <td>8.79</td>
      <td>3.85</td>
      <td>1</td>
      <td>erc</td>
      <td>20</td>
      <td>20</td>
      <td>1</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>20</td>
      <td>24.08</td>
      <td>2.26</td>
      <td>1</td>
      <td>gc</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# label
# what was actually chosen.
label = df[df['depvar'] == True].sort_values(by='idx.id1')['idx.id2'].reset_index(drop=True)
item_names = ['ec', 'ecc', 'er', 'erc', 'gc', 'gcc', 'hpc']
num_items = df['idx.id2'].nunique()
# cardinal encoder.
encoder = dict(zip(item_names, range(num_items)))
label = label.map(lambda x: encoder[x])
label = torch.LongTensor(label)
```


```python
# category feature: no category feature, all features are item-level.
category_dataset = ChoiceDataset(label=label.clone()).to(device)

# item feature.
item_feat_cols = ['ich', 'och', 'icca', 'occa', 'inc.room', 'inc.cooling', 'int.cooling']
price_obs = utils.pivot3d(df, dim0='idx.id1', dim1='idx.id2', values=item_feat_cols)
item_dataset = ChoiceDataset(label=label, price_obs=price_obs).to(device)

# combine dataets
dataset = JointDataset(category=category_dataset, item=item_dataset)
# data_loader = utils.create_data_loader(dataset)
```

# Example 1
Run a nested logit model on the data for two nests and one log-sum coefficient that applies to both nests. Note that the model is specified to have the cooling alternatives `{gcc, ecc, erc, hpc}` in one nest and the non-cooling alternatives `{gc, ec, er}` in another nest.


```python
category_to_item = {0: ['gcc', 'ecc', 'erc', 'hpc'],
                    1: ['gc', 'ec', 'er']}

# convert names
for k, v in category_to_item.items():
    v = [encoder[item] for item in v]
    category_to_item[k] = sorted(v)

model = NestedLogitModel(category_to_item=category_to_item,
                         category_coef_variation_dict={},
                         category_num_param_dict={},
                         item_coef_variation_dict={'price_obs': 'constant'},
                         item_num_param_dict={'price_obs': 7},
                         shared_lambda=True)

model = model.to(device)
```

**NOTE**: We are computing the standard errors using $\sqrt{\text{diag}(H^{-1})}$, where $H$ is the
hessian of negative log-likelihood with respect to model parameters. This leads to slight different results compared with R implementation.

```python
run(model, dataset, num_epochs=10000)
```

    ==================== received model ====================
    NestedLogitModel(
      (category_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total).
      )
    )
    ==================== received dataset ====================
    JointDataset with 2 sub-datasets: (
    	category: ChoiceDataset(label=[250], user_index=[], session_index=[], item_availability=[], observable_prefix=[5], device=cuda:0)
    	item: ChoiceDataset(label=[250], user_index=[], session_index=[250], item_availability=[], observable_prefix=[5], price_obs=[250, 7, 7], device=cuda:0)
    )
    ==================== training the model ====================
    Epoch 1000: Mean Log-likelihood=-211.021728515625
    Epoch 2000: Mean Log-likelihood=-193.19943237304688
    Epoch 3000: Mean Log-likelihood=-182.55709838867188
    Epoch 4000: Mean Log-likelihood=-179.28445434570312
    Epoch 5000: Mean Log-likelihood=-178.64239501953125
    Epoch 6000: Mean Log-likelihood=-178.47711181640625
    Epoch 7000: Mean Log-likelihood=-178.3613739013672
    Epoch 8000: Mean Log-likelihood=-178.25039672851562
    Epoch 9000: Mean Log-likelihood=-178.1700439453125
    Epoch 10000: Mean Log-likelihood=-178.1334686279297
    ==================== model results ====================
    Training Epochs: 10000

    Learning Rate: 0.01

    Batch Size: 250 out of 250 observations in total

    Final Log-likelihood: -178.1334686279297

    Coefficients:

    | Coefficient      |   Estimation |   Std. Err. |
    |:-----------------|-------------:|------------:|
    | lambda_weight_0  |     0.577392 |   0.164925  |
    | item_price_obs_0 |    -0.546838 |   0.14313   |
    | item_price_obs_1 |    -0.845657 |   0.23527   |
    | item_price_obs_2 |    -0.234306 |   0.110563  |
    | item_price_obs_3 |    -1.18538  |   1.03586   |
    | item_price_obs_4 |    -0.373288 |   0.0996275 |
    | item_price_obs_5 |     0.248309 |   0.051553  |
    | item_price_obs_6 |    -5.36669  |   4.76851   |


## R Output
```
##
## Call:
## mlogit(formula = depvar ~ ich + och + icca + occa + inc.room +
##     inc.cooling + int.cooling | 0, data = HC, nests = list(cooling = c("gcc",
##     "ecc", "erc", "hpc"), other = c("gc", "ec", "er")), un.nest.el = TRUE)
##
## Frequencies of alternatives:choice
##    ec   ecc    er   erc    gc   gcc   hpc
## 0.004 0.016 0.032 0.004 0.096 0.744 0.104
##
## bfgs method
## 11 iterations, 0h:0m:0s
## g'(-H)^-1g = 7.26E-06
## successive function values within tolerance limits
##
## Coefficients :
##              Estimate Std. Error z-value  Pr(>|z|)
## ich         -0.554878   0.144205 -3.8478 0.0001192 ***
## och         -0.857886   0.255313 -3.3601 0.0007791 ***
## icca        -0.225079   0.144423 -1.5585 0.1191212
## occa        -1.089458   1.219821 -0.8931 0.3717882
## inc.room    -0.378971   0.099631 -3.8038 0.0001425 ***
## inc.cooling  0.249575   0.059213  4.2149 2.499e-05 ***
## int.cooling -6.000415   5.562423 -1.0787 0.2807030
## iv           0.585922   0.179708  3.2604 0.0011125 **
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
##
## Log-Likelihood: -178.12
```


# Example 2
Re-estimate the model with the room alternatives in one nest and the central alternatives in another nest. (Note that a heat pump is a central system.)


```python
category_to_item = {0: ['ec', 'ecc', 'gc', 'gcc', 'hpc'],
                    1: ['er', 'erc']}
for k, v in category_to_item.items():
    v = [encoder[item] for item in v]
    category_to_item[k] = sorted(v)

model = NestedLogitModel(category_to_item=category_to_item,
                            category_coef_variation_dict={},
                            category_num_param_dict={},
                        #  category_num_param_dict={'intercept': 1},
                            item_coef_variation_dict={'price_obs': 'constant'},
                            item_num_param_dict={'price_obs': 7},
                            shared_lambda=True
                            )

model = model.to(device)
```


```python
run(model, dataset, num_epochs=5000, learning_rate=0.3)
```

    ==================== received model ====================
    NestedLogitModel(
      (category_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total).
      )
    )
    ==================== received dataset ====================
    JointDataset with 2 sub-datasets: (
    	category: ChoiceDataset(label=[250], user_index=[], session_index=[], item_availability=[], observable_prefix=[5], device=cuda:0)
    	item: ChoiceDataset(label=[250], user_index=[], session_index=[250], item_availability=[], observable_prefix=[5], price_obs=[250, 7, 7], device=cuda:0)
    )
    ==================== training the model ====================
    Epoch 500: Mean Log-likelihood=-189.6171875
    Epoch 1000: Mean Log-likelihood=-183.52847290039062
    Epoch 1500: Mean Log-likelihood=-181.83969116210938
    Epoch 2000: Mean Log-likelihood=-180.43809509277344
    Epoch 2500: Mean Log-likelihood=-180.04177856445312
    Epoch 3000: Mean Log-likelihood=-180.17776489257812
    Epoch 3500: Mean Log-likelihood=-180.0731964111328
    Epoch 4000: Mean Log-likelihood=-180.2925567626953
    Epoch 4500: Mean Log-likelihood=-180.3995361328125
    Epoch 5000: Mean Log-likelihood=-180.7696533203125
    ==================== model results ====================
    Training Epochs: 5000

    Learning Rate: 0.3

    Batch Size: 250 out of 250 observations in total

    Final Log-likelihood: -180.7696533203125

    Coefficients:

    | Coefficient      |   Estimation |   Std. Err. |
    |:-----------------|-------------:|------------:|
    | lambda_weight_0  |     1.59468  |    0.755122 |
    | item_price_obs_0 |    -1.33103  |    0.602767 |
    | item_price_obs_1 |    -2.13653  |    0.999629 |
    | item_price_obs_2 |    -0.380562 |    0.243608 |
    | item_price_obs_3 |    -2.55752  |    2.22184  |
    | item_price_obs_4 |    -0.875605 |    0.365037 |
    | item_price_obs_5 |     0.493142 |    0.238891 |
    | item_price_obs_6 |   -15.6913   |    9.80992  |


## R Output
```
##
## Call:
## mlogit(formula = depvar ~ ich + och + icca + occa + inc.room +
##     inc.cooling + int.cooling | 0, data = HC, nests = list(central = c("ec",
##     "ecc", "gc", "gcc", "hpc"), room = c("er", "erc")), un.nest.el = TRUE)
##
## Frequencies of alternatives:choice
##    ec   ecc    er   erc    gc   gcc   hpc
## 0.004 0.016 0.032 0.004 0.096 0.744 0.104
##
## bfgs method
## 10 iterations, 0h:0m:0s
## g'(-H)^-1g = 5.87E-07
## gradient close to zero
##
## Coefficients :
##              Estimate Std. Error z-value Pr(>|z|)
## ich          -1.13818    0.54216 -2.0993  0.03579 *
## och          -1.82532    0.93228 -1.9579  0.05024 .
## icca         -0.33746    0.26934 -1.2529  0.21024
## occa         -2.06328    1.89726 -1.0875  0.27681
## inc.room     -0.75722    0.34292 -2.2081  0.02723 *
## inc.cooling   0.41689    0.20742  2.0099  0.04444 *
## int.cooling -13.82487    7.94031 -1.7411  0.08167 .
## iv            1.36201    0.65393  2.0828  0.03727 *
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
##
## Log-Likelihood: -180.02

```

# Example 3
Rewrite the code to allow three nests. For simplicity, estimate only one log-sum coefficient which is applied to all three nests. Estimate a model with alternatives gcc, ecc and erc in a nest, hpc in a nest alone, and alternatives gc, ec and er in a nest. Does this model seem better or worse than the model in exercise 1, which puts alternative hpc in the same nest as alternatives gcc, ecc and erc?


```python
category_to_item = {0: ['gcc', 'ecc', 'erc'],
                    1: ['hpc'],
                    2: ['gc', 'ec', 'er']}
for k, v in category_to_item.items():
    v = [encoder[item] for item in v]
    category_to_item[k] = sorted(v)

model = NestedLogitModel(category_to_item=category_to_item,
                         category_coef_variation_dict={},
                         category_num_param_dict={},
                         item_coef_variation_dict={'price_obs': 'constant'},
                         item_num_param_dict={'price_obs': 7},
                         shared_lambda=True
                         )

model = model.to(device)
```


```python
run(model, dataset)
```

    ==================== received model ====================
    NestedLogitModel(
      (category_coef_dict): ModuleDict()
      (item_coef_dict): ModuleDict(
        (price_obs): Coefficient(variation=constant, num_items=7, num_users=None, num_params=7, 7 trainable parameters in total).
      )
    )
    ==================== received dataset ====================
    JointDataset with 2 sub-datasets: (
    	category: ChoiceDataset(label=[250], user_index=[], session_index=[], item_availability=[], observable_prefix=[5], device=cuda:0)
    	item: ChoiceDataset(label=[250], user_index=[], session_index=[250], item_availability=[], observable_prefix=[5], price_obs=[250, 7, 7], device=cuda:0)
    )
    ==================== training the model ====================
    Epoch 500: Mean Log-likelihood=-186.1810302734375
    Epoch 1000: Mean Log-likelihood=-182.77426147460938
    Epoch 1500: Mean Log-likelihood=-181.76400756835938
    Epoch 2000: Mean Log-likelihood=-181.4237060546875
    Epoch 2500: Mean Log-likelihood=-181.26036071777344
    Epoch 3000: Mean Log-likelihood=-181.11245727539062
    Epoch 3500: Mean Log-likelihood=-180.95578002929688
    Epoch 4000: Mean Log-likelihood=-180.79638671875
    Epoch 4500: Mean Log-likelihood=-180.64500427246094
    Epoch 5000: Mean Log-likelihood=-180.5127410888672
    ==================== model results ====================
    Training Epochs: 5000

    Learning Rate: 0.01

    Batch Size: 250 out of 250 observations in total

    Final Log-likelihood: -180.5127410888672

    Coefficients:

    | Coefficient      |   Estimation |   Std. Err. |
    |:-----------------|-------------:|------------:|
    | lambda_weight_0  |     0.934802 |   0.19279   |
    | item_price_obs_0 |    -0.819485 |   0.0967955 |
    | item_price_obs_1 |    -1.30911  |   0.182129  |
    | item_price_obs_2 |    -0.321997 |   0.127624  |
    | item_price_obs_3 |    -2.05462  |   1.15039   |
    | item_price_obs_4 |    -0.55606  |   0.0729355 |
    | item_price_obs_5 |     0.310064 |   0.0552311 |
    | item_price_obs_6 |    -6.78603  |   5.06669   |


## R Output
```
##
## Call:
## mlogit(formula = depvar ~ ich + och + icca + occa + inc.room +
##     inc.cooling + int.cooling | 0, data = HC, nests = list(n1 = c("gcc",
##     "ecc", "erc"), n2 = c("hpc"), n3 = c("gc", "ec", "er")),
##     un.nest.el = TRUE)
##
## Frequencies of alternatives:choice
##    ec   ecc    er   erc    gc   gcc   hpc
## 0.004 0.016 0.032 0.004 0.096 0.744 0.104
##
## bfgs method
## 8 iterations, 0h:0m:0s
## g'(-H)^-1g = 3.71E-08
## gradient close to zero
##
## Coefficients :
##               Estimate Std. Error z-value  Pr(>|z|)
## ich          -0.838394   0.100546 -8.3384 < 2.2e-16 ***
## och          -1.331598   0.252069 -5.2827 1.273e-07 ***
## icca         -0.256131   0.145564 -1.7596   0.07848 .
## occa         -1.405656   1.207281 -1.1643   0.24430
## inc.room     -0.571352   0.077950 -7.3297 2.307e-13 ***
## inc.cooling   0.311355   0.056357  5.5247 3.301e-08 ***
## int.cooling -10.413384   5.612445 -1.8554   0.06354 .
## iv            0.956544   0.180722  5.2929 1.204e-07 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
##
## Log-Likelihood: -180.26
```

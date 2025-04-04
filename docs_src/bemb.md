<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
# BEMB Tutorial
This tutorial assumes the reader has ready gone through the [Data Management](./data_management.md) tutorial.
Through this tutorial, we use Greek letters (except for $\varepsilon$ as error term) to denote learnable coefficients of the model. However, researchers are not restricted to use Greek letters in practice.

Bayesian EMBedding (BEMB) is a hierarchical Bayesian model for modelling consumer choices.
The model can naturally extend to other use cases which can be formulated into the consumer choice framework.
For example, in a job-transition modelling study, we formulated the starting job as the user and ending job as the item and applied the BEMB framework.

## The BEMB Model
This section offers some technical background of the BEMB model

The BEMB model is a general Bayesian version of random utility models with independence of irrelevant alternatives.

Suppose we have a dataset of purchase records from $U$ users, $I$ items, and $S$ sessions (the [introduction tutorial](https://gsbdbi.github.io/torch-choice/intro/) helps you could revise these terminologies).
Further, the set items can be partitioned into $C$ **categories** indexed by $c \in \{1,2,\dots,C\}$. Let $I_c$ denote the collection of items in category $c$. It's easy to see that the union of all $I_c$ is the entire set of items $\{1, 2, \dots I\}$.

The model assumes unit demand for each category, independent choices across categories.

We aim to capture the utility of user $u$ from purchasing item $i$ in the context of session $s$, denoted as $\mathcal{U}_{uis}$.
The utility can be decomposed into a deterministic utility $U_{uis}$ and an error term $\varepsilon_{uis}$ following the Gumbel distribution (logit):

$$
\mathcal{U}_{uis} = U_{uis} + \varepsilon_{uis}
$$

The `bemb` package allows researchers to specify various formulations of the deterministic term $U_{uis}$, for example:

$$
U_{uis} = \alpha + \beta^\top \gamma + \delta \times X + (\eta^\top \zeta) \times Z
$$

Our package allows for four kinds of additive terms in $U_{uis}$.
In the example above, Greek letters denote the learnable coefficients of the model.
Coefficients are versatile: they can be constant, user-specific, or item-specific. $X$ and $Z$ denote observables.

The package allows for four types of additive term in $U_{uis}$:
1. intercept terms like $\alpha$;
2. interaction terms like $\beta^\top \gamma$;
3. observable terms like $\delta \times X$;
4. observable terms with interactive coefficient like $(\eta^\top \zeta) \times Z$.

Finally, the model predicts the probability of user $u$ choosing item $i$ among items in the same category $I_c$ in the context of session $s$ as:

$$
P(i|u,s) = \frac{e^{U(u, i, s)}}{\sum_{i' \in I_c} e^{U(u, i', s)}}
$$

For example, at BEMB's core (assume no $s$-level effect for now), the BEMB model aims to build user embeddings $\theta_u \in \mathbb{R}^{L}$ and item embeddings $\alpha_i \in \mathbb{R}^{L}$. The utility of user $u$ from purchasing item $i$ is $U_{ui} = \theta_u^\top \alpha_i$, and the model predicts the probability for user $u$ to purchase item $i$ as an increasing function of $U_{ui}$.
Our package support more general form of utility $U_{ui}$ than the inner product of two latent vectors.

Both of $\theta_u$ and $\alpha_i$ are *Bayesian*, which means there is a prior distribution and a variational distribution associated with each of them.
By default, the prior distribution of all entries of $\theta_u$ and $\alpha_i$ are i.i.d. standard Gaussian distributions.
The variational distributions are Gaussian with learnable mean and standard deviation, these parameters are trained by minimizing the ELBO so that the predicted purchasing probabilities best fit the observed dataset.

**TODO**: add reference to the paper introducing BEMB for a complete description of the model.

## Running BEMB
Running BEMB requires you to (1) build the `ChoiceDataset` object and (2) training the model.
## The `ChoiceDataset`
Please refer to the [Data Management](./data_management.md) tutorial for a detailed walk-through of how to constructing the dataset.
For simplicity, we assume that item/user/session/price observables are named as `{item, user, session, price}_obs` in the `ChoiceDataset`, the researcher can use arbitrary variable names as long as they satisfy the naming convention (e.g., user-level observables should start with `user_` and cannot be `user_index`) and have the correct shape (e.g., user-level observables should have shape `(num_users, num_obs)`).

## Setup the BEMB Model (PyTorch-Lightning Interface)

You will be constructing the `LitBEMBFlex` class, which is a PyTorch-lightning wrapper of the BEMB model implemented in plain PyTorch. The lighting wrapper free researchers from complications such as setting up the training loop and optimizers.

To initialize the `LitBEMBFlex` class, the researcher needs to provide it with the following arguments. We recommend the research to encompass all arguments in a separate yaml file. Most of these arguments should be self explanatory, Please refer to the doc string of `BEMBFlex.__init__()` for a detailed elaboration.

### Utility Formula: `utility_formula`
**Note**: for the string parsing to work correctly, please **do** add spaces around `+` and `*`.
This section covers how to convert the utility representation in a choice problem into the `utility_formula` argument of the `BEMBFlex` model and `LitBEMBFlex` wrapper.

The core of specifying a BEMB model is to **specify the utility function** $U(u,i,s)$ for user $u$ to purchase item $i$ in session $s$, the `bemb` package provides an easy-to-use string-parsing mechanism for researchers to provide their ideal utility representations.
With the utility representation, the probability for consumer $u$ to purchase item $i$ in session $s$ is the following

$$
P(i|u,s) = \frac{e^{U(u, i, s)}}{\sum_{i' \in I_c} e^{U(u, i', s)}}
$$

where $I_c$ is the set of items in the same category of item $i$.
If there is no category information, the model considers all items to be in the same category, i.e., $I_c = \{1, 2, \dots I\}$.

The BEMB admits a **linear additive form** of utility formula.
This is a very flexible formulation because
1. you can always build sophisticated observables, for instance by taking the log or a polynomial transformation of original observables and
2. you can impose that the learnable coefficients depend on item $i$, user $u$, session $s$ or the combination of item and session.

For example, the model parses utility formula string `lambda_item + theta_user * alpha_item + zeta_user * item_obs` into the following representation:

$$
U(u, i, s)= \lambda_i + \theta_u^\top \alpha_i + \zeta_u^\top X^{item}_i + \varepsilon_{uis} \in \mathbb{R}
$$

The `utility_formula` consists of two classes of objects:
1. **learnable coefficients** (i.e., Greek letters): the string-parser identifies learnable coefficients by looking at their suffix. These variables can be (1) constant across all items and users, (2) user-specific, or (3) item-specific. For example, the $\lambda_i$ term above is item-specific intercept and it is presented as `item_item` in the `utility_formula`. To ensure the string-parsing is working properly, learnable coefficients **must** ends with one of `{_item, _user, _constant}`.
2. **Observable Tensors** are identified by their prefix, which tells whether they are item-specific (with `item_` prefix), user-specific (with `user_` prefix), session-specific (with `session_` prefix), or session-and-item-specific (with `price_` prefix) observables. Each of these observables should present in the `ChoiceDataset` data structure constructed.

**Warning**: the `utility_formula` parser identifies learnable coefficients as using suffix and observables using prefix, the researcher should **never** name things with both prefix in `{user_, item_, session_, price_}` and suffix `{_constant, _user, _item}` such as `item_quality_user`.

Overall, there are four types of additive component, except the error term $\epsilon$, in the utility representation:

1. Standalone coefficients $\lambda, \lambda_i, \lambda_u \in \mathbb{R}$ representing intercepts and item/user level fixed effects.
2. “Matrix factorization” coefficients $\theta_u^\top \alpha_i$, where $\theta_u,\alpha_i \in \mathbb{R}^L$ are embedding/latent of users and items, $L$ is the latent dimension specified by the researcher.
3. Observable terms $\zeta_u^\top X^{item}_i$, where each $\zeta_u \in \mathbb{R}^{K_{item}}$ is the user specific coefficients for item observables. This type of component is written as `zeta_user * item_obs` in the utility formula. For sure, one can use coefficients constant among users by simply putting `zeta_constant` in the utility formula.
4.  “Matrix factorization” coefficients of observables written as `gamma_user * beta_item * price_obs`.  This type of component factorizes the coefficient of observables into user and item latents. For example, suppose there are $K_{price}$ price observables (i.e., observables varying by both item and session, price is one of them!), for each of price observable $X^{price}_{is}[k] \in \mathbb{R}$, a pair of latent $\gamma_u^k, \beta_i^k \in \mathbb{R}^L$ is trained to construct the coefficient of the $k^{th}$ price observable, where $L$ is the latent dimension specified by the researcher. In this case, the utility is  $U(u, i, s) = \sum_{k=1}^K (\gamma_u^{k\top} \beta_i^k) X^{price}_{is}[k]$. One can for sure replace the `price_obs` with any of `{user, item, session}_obs`.

If the researcher wish to treat different part of item observable differently, for example,

$$
U(u, i, s) = \dots + \zeta_u^\top Y^{item}_i + \omega^\top Z^{item}_i + \dots
$$

where we partition item observables into two parts $X^{item}_i = [Y^{item}_i, Z^{item}_i]$, and the coefficient for $Y^{item}_i$ is user-specific but the coefficient for the second part is constant across all users.
In this case, the researcher should use separate tensors `item_obs_part1` and `item_obs_part2` while constructing the `ChoiceDataset` (both of them needs to start with `item_`), and use the `utility formula` with `zeta_user * item_obs_part1 + omega_constant * item_obs_part2`.

With the above four cases as building blocks, the researcher can specify all kinds of utility functions.

### Number of Users/Items/Sessions `num_{users, items, sessions}`
The researcher is responsible for providing the size of the prediction problem.
For every model, the `num_items` is **required**.
However, `num_users` and `num_sessions` are required only if there is any user/session-specific observables or parameters involved in the `utility_formula`.

### Specifying the Dimensions of Coefficients with the `coef_dim_dict` dictionary
To correctly initialize the model, the constructor needs to know the shape of each learnable coefficients (i.e., Greek letters above). For item/user-specific parameters, the value of `coef_dim_dict` is the number of parameters for **each** user/item, not the total number of parameters.

#### 1. Intercept Terms
For standalone coefficients like `lambda_item`, `coef_dim_dict['lambda_item']` = 1 always.

#### 2. Matrix Factorization Terms
For matrix factorization coefficients like `theta_user` and `alpha_item`, `coef_dim_dict['theta_user'] = coef_dim_dict[alpha_item] = L`, where `L` is the desired latent dimension. For the inner product between $\alpha_i$ and $\theta_u$ to work properly, `coef_dim_dict['theta_user'] == coef_dim_dict['alpha_item']`.

#### 3. Coefficient and Observable Terms
For terms like $\zeta_u^\top X^{item}_i$, `coef_dim_dict['zeta_user']` needs to be the dimension of $X^{item}_i$.

#### 4. Factorized Coefficient and Observable Terms
For the most complicated matrix factorization coefficients, the dimension needs to be the latent dimension multiplied by the number of observables. For example, if you have a $K$-dimensional feature vector $\textbf{x} \in \mathbb{R}^K$, and the utility contains an additive component $\zeta_{u, i}^\top \textbf{x} \in \mathbb{R}$. The coefficient $\zeta_{u, i} \in \mathbb{R}^K$ comes from of the user-specific and item-specific parts so that the coefficient depends on both the user and the item.
The coefficient $\zeta_{ui}$ is defined as

$$
\zeta_{ui} = \begin{pmatrix} \gamma_{u, 1}^\top \beta_{i, 1} \\ \gamma_{u,2}^\top \beta_{i, 2} \\ \vdots \\ \gamma_{u, K}^\top \beta_{i, K} \end{pmatrix} \in \mathbb{R}^K
$$

for each $k \in \{1, 2, \dots, K\}$, $\gamma_{u, k}, \beta_{i, k}$ is a $L$-dimensional vector, the dimension $L$ is called the **latent dimension**.

Let's use superscript $(k)$ to denote the $k^{th}$ component of a vector, the decomposition works as the following:

$$
\zeta_{u, i}^\top \textbf{x} = \sum_{k=1}^{K} \underbrace{\zeta_{u, i}^{(k)}}_{\in \mathbb{R}} \underbrace{\textbf{x}^{(k)}}_{\in \mathbb{R}}
= \sum_{k=1}^{K} \underbrace{(\gamma_{u, k}^\top \beta_{i, k})}_{\in \mathbb{R}} \underbrace{\textbf{x}^{(k)}}_{\in \mathbb{R}}
$$

Equivalently and more succinctly, if we define matrices by concatenating the vectors,

$$
\gamma_u =
\begin{pmatrix}
-\gamma_{u, 1}-  \\
-\gamma_{u, 2}-  \\
\vdots \\
-\gamma_{u, K}-  \\
\end{pmatrix}
,\quad
\beta_i =
\begin{pmatrix}
-\beta_{i, 1}-\\
-\beta_{i, 2}-\\
\vdots \\
-\beta_{i, K}-\\
\end{pmatrix}
$$

both $\gamma_u, \beta_i \in \mathbb{R}^{K \times L}$, it's immediate that for each user $u$ or item $i$, we have $K \times L$ learnable parameters/coefficients. Therefore, the `coef_dim_dict[gamma_user] = coef_dim_dict[beta_item] = K * L`.

For example, suppose we have a two dimensional observable $\textbf{x} = (x_1, x_2)$ (so $K = 2$), and we wish to include a term $\zeta_{ui}^\top \textbf{x}$ in the utility formula: $\zeta_{ui}^{(1)} \times x_1 + \zeta_{ui}^{(2)} \times x_2$.

Further, we want to decompose each of $\zeta_{ui}^{(1)} \in \mathbb{R}$ and $\zeta_{ui}^{(2)} \in \mathbb{R}$ into a 10-dimensional user component and 10-dimensional item component (so $L = 10$): $\zeta_{ui}^{(1)} = \gamma_{u, 1}^\top \beta_{i, 1}$, with $\gamma_{u, 1}, \beta_{i, 1} \in \mathbb{R}^{10}$ (similarly, $\zeta_{ui}^{(2)} = \gamma_{u, 2}^\top \beta_{i, 2}$). In this example, we need to set `coef_dim_dict[gamma_user] = coef_dim_dict[beta_item] = 30`.

**Note**: since we need to compute the inner product between $\gamma$'s and $\beta$'s, the `coef_dim_dict[gamma_user]` and `coef_dim_dict[beta_item]` need to be the same.

**Upcoming Updates**: sounds like a lot of work? we are currently developing helper function to infer all these information from the `ChoiceDataset`, but we will still provide researchers with the full control over the configuration.

### Variance of Prior Distributions: Specifying Variance of Coefficient Prior Distributions with `prior_variance`
The `bemb` package allows for specifying the variance of prior distributions for learnable parameters.

The `prior_variance` term can be either a scalar or a dictionary with the same keys of `coef_dim_dict`, which provides the variance of prior distribution for each learnable coefficients.
If a float is provided, all priors will be Gaussian distribution with diagonal covariance matrix with `prior_variance` along the diagonal.
If a dictionary is provided, keys of `prior_variance` should be coefficient names, and the prior of each `coef_name` would be a Gaussian with diagonal covariance matrix with `prior_variance[coef_name]` along the diagonal.
This value is default to be `1.0`, which means priors of all coefficients are standard Gaussian distributions.

### Expectations of Prior Distributions: Incorporating Observables to the Bayesian Prior with `obs2prior_dict`
BEMB is a Bayesian factorization model trained by optimizing the evidence lower bound (ELBO). Each parameter (i.e., these with `_item, _user, _constant` suffix.) in the BEMB model carries a prior distribution, which is set to $\mathcal{N}(\mathbf{0}, Var)$.
The variance of prior distribution is governed by the `prior_variance` term mentioned above.

Beyond this baseline case, the hierarchical nature of BEMB allows the mean of the prior distribution to depend on observables as a (learnable) linear mapping. For example:

$$
\theta_{i} \overset{prior}{\sim} \mathcal{N}(HX^{item}_i, Var)
$$

where the prior mean is a linear transformation of the item observable and $H: \mathbb{R}^{K_{item}} \to \mathbb{R}^L$.

**Note**: the exact form of prior variance $Var$ depends on the `prior_variance` specified. The default $Var$ the identity matrix $I$, so that all entries of coefficients are independent and have unit variance.

To enable the observable-to-prior feature, one needs to set `obs2prior_dict['theta_item']=True`.
In order to leverage obs-to-prior for item-specific coefficients like `theta_item`, the researchers need to include `item_obs` tensor to the `ChoiceDataset`, *the attribute name needs to be exactly `item_obs`, just with `item_` prefix is **not** sufficient.* Similarly, `user_obs` are required if obs-to-prior is turned on for **any** of user-specific coefficients.

Please see the [dedicated `obs2prior` tutorial](https://gsbdbi.github.io/bemb/bemb_obs2prior_simulation/) for more details.

#### Advanced Usage: Zeroing Out Entries of $H$ Matrix
In certain cases, one wishes to enforce sparsity of the $H$ matrix, which is possible by setting the `H_zero_mask_dict`.
For example, this feature is particularly useful in cases when the researcher want the first coordinate of latent prior to be independent from certain observables, which can be achieved by setting corresponding entries in $H$ to be zero.

The `H_zero_mask_dict` is a dictionary with the same keys of `coef_dim_dict`, and the value of each key is a boolean matrix with the same shape of $H$ indicating whether the corresponding $H$ entry is zeroed out.

Please refer to [this tutorial](https://github.com/gsbDBI/bemb/blob/main/tutorials/simulation_H_zero_mask/simulation_H_zero_mask.ipynb) for more details of this feature.

**Summary**: the `prior_variance` term controls the variance of prior distribution and `obs2prior` term controls the expectation of prior distribution.

### Grouping Items into Categories with `category_to_item`
By default, the probability of purchasing item $i$ by user $u$ in session $s$ is, where the summation in the denominator is over all items $i$:

$$
P(i|u,s) = \frac{e^{U(u, i, s)}}{\sum_{i'=1}^I e^{U(u, i', s)}}
$$

In some cases, the researcher wishes to provide additional guidance to the model by providing the category of the bought item in teach purchasing record.
In this case, the probability of purchasing each $i$ will be normalized only across other items from the same category rather than all items.

The `category_to_item` argument provides a dictionary with category id or name as keys, and `category_to_item[C]` contains the list of item ids belonging to category `C`.

With `category_to_item` provided, for the probability of purchasing item $i$ by user $u$ in session, let $I_c$ denote the set of items belonging to the same category $i$, the probability of purchasing is (note the difference in summation scope, it's over items in the same category as $i$ only):

$$
P(i|u,s) = \frac{e^{U(u, i, s)}}{\sum_{i' \in I_c} e^{U(u, i', s)}}
$$

### Last Step: Create the `LitBEMBFlex` wrapper
The last step is to create the `LitBEMBFlex` object which contains all information we gathered above. You will also need to provide a `learning_rate` for the the optimizer and a `num_seeds` for the Monte-Carlo estimator of gradient in ELBO.
```python
model = bemb.model.LitBEMBFlex(
    learning_rate=0.01,
    num_seeds=4,
    utility_formula=utility_formula,
    num_users=num_users,
    num_items=num_items,
    num_sessions=num_sessions,
    obs2prior_dict=obs2prior_dict,
    coef_dim_dict=coef_dim_dict,
    category_to_item=category_to_item,
    num_user_obs=num_user_obs,
    num_item_obs=num_item_obs,
)
```

## Training the Model
We provide a ready-to-use scrip to train the model, where `dataset_list` is a list consists of three `ChoiceDataset` objects (see the [Data Management](./data_management.md) tutorial for splitting datasets), the training, validation, and testing dataset.
```python
model = model.to('cuda')  # only if GPU is installed
model = bemb.utils.run_helper.run(model, dataset_list, batch_size=32, num_epochs=10)
```

## Inference
Lastly, to get the utilities (i.e., the logit) of the item bought for each row of the test dataset, the `model.model.forward()` method does the trick.
You can either compute the utility for the purchased item or for all items, we put significant effort on optimizing the pipeline for estimating utility of the bought item, so it's much faster than computing utilities of all items.
```python
with torch.no_grad():
    # disable gradient tracking to save computational cost.
    utility_chosen = model.model.forward(dataset_list[2], return_logit=True, all_items=False)
    # uses much higher memory!
    utility_all = model.model.forward(dataset_list[2], return_logit=True, all_items=True)
```

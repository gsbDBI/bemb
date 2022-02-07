<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
# BEMB Tutorial
This tutorial assumes the reader has ready gone through the [Data Management](./data_management.md) tutorial.
Through this tutorial, we use Greek letters (except for $$\varepsilon$$ as error term) to denote learnable parameters of the model.

Bayesian EMBedding (BEMB) is a hierarchical Bayesian model for modelling consumer choices.
The model can naturally extend to other use cases which can be formulated into the consumer choice framework.
For example, in a job-transition modelling study, we formulated the starting job as the user and ending job as the item and applied the BEMB framework.
Suppose we have a dataset of purchase records consisting of $$U$$ users, $$I$$ items, and $$S$$ sessions, at it's core (assume no $$s$$-level effect for now), the BEMB model aims to build user embeddings $$\theta_u \in \mathbb{R}^{L}$$ and item embeddings $$\alpha_i \in \mathbb{R}^{L}$$, then the model predicts the probability for user $$u$$ to purchase item $$i$$ as
$$
P(i|u,s) \propto \theta_u^\top \alpha_i.
$$
Both of $$\theta_u$$ and $$\alpha_i$$ are *Bayesian*, which means there is a prior distribution and a variational distribution associated with each of them.
By default, the prior distribution of all entries of $$\theta_u$$ and $$\alpha_i$$ are i.i.d. standard Gaussian distributions.
The variational distributions are Gaussian with learnable mean and standard deviation, these parameters are trained by minimizing the ELBO so that the predicted purchasing probabilities best fit the observed dataset.

**TODO**: add reference to the paper introducing BEMB for a complete description of the model.

## Running BEMB
Running BEMB requires you to (1) build the `ChoiceDataset` object and (2) training the model.
## The `ChoiceDataset`
Please refer to the [Data Management](./data_management.md) tutorial for a detailed walk-through of how to constructing the dataset.
For simplicity, we assume that item/user/session/price observables are named as `{item, user, session, price}_obs` in the `ChoiceDataset`, the researcher can use arbitrary variable names as long as they satisfy the naming convention (e.g., user-level observables should start with `user_` and cannot be `user_index`) and have the correct shape (e.g., user-level observables should have shape `(num_users, num_obs)`).

## Setup the BEMB Model (PyTorch-Lightning Interface)

You will be constructing the `LitBEMBFlex` class, which is a PyTorch-lightning wrapper of the BEMB model implemented in plain PyTorch. The lighting wrapper free researchers from complications such as setting up the training loop and optimizers.

To initialize the `LitBEMBFlex` class, the researcher needs to provide it with the following components. We recommend the research to encompass all arguments in a separate yaml file. Most of these arguments should be self explanatory, Please refer to the doc string of `BEMBFlex.__init__()` for a detailed elaboration.

### Utility Formula `utility_formula`
**Note**: for the string parsing to work correctly, please **do** add spaces around `+` and `*`.
This section covers how to convert the utility representation in a choice problem into the `utility_formula` argument of the `BEMBFlex` model and `LitBEMBFlex` wrapper.

The core of specifying a BEMB model is to **specify the utility function** $$\mathcal{U}(u,i,s)$$ for user $$u$$ to purchase item $$i$$ in session $$s$$, the `bemb` package provides an easy-to-use string-parsing mechanism for researchers to provide their ideal utility representations.
With the utility representation, the probability for consumer $$u$$ to purchase item $$i$$ in session $$s$$ is the following
$$
P(i|u,s) = \frac{e^{\mathcal{U}(u, i, s)}}{\sum_{i' \in I_c} e^{\mathcal{U}(u, i', s)}}
$$
where $$I_c$$ is the set of items in the same category of item $$i$$.
If there is no category information, the model considers all items to be in the same category, i.e., $$I_c = \{1, 2, \dots I}$$.

The BEMB admits a **linear additive form** of utility formula.

For example, the model parses utility formula string `lambda_item + theta_user * alpha_item + zeta_user * item_obs` into the following representation:

$$
\mathcal{U}(u, i, s)= \lambda_i + \theta_u^\top \alpha_i + \zeta_u^\top X^{item}_i \varepsilon \in \mathbb{R}
$$

The `utility_formula` consists of two classes of objects:
1. **Learnable Parameters** (i.e., Greek letters): the string-parser identifies learnable parameters by looking at their suffix. These variables can be (1) constant across all items and users, (2) user-specific, or (3) item-specific. For example, the $$\lambda_i$$ term above is item-specific intercept and it is presented as `item_item` in the `utility_formula`. To ensure the string-parsing is working properly, learnable parameters **must** ends with one of `{_item, _user, _constant}`.
2. **Observable Tensors** are identified by their prefix, which tells whether they are item-specific (with `item_` prefix), user-specific (with `user_` prefix), session-specific (with `session_` prefix), or session-and-item-specific (with `price_` prefix) observables. Each of these observables should present in the `ChoiceDataset` data structure constructed.

**Warning**: the `utility_formula` parser identifies learnable parameters as using suffix and observables using prefix, the researcher should **never** name things with both prefix in `{user_, item_, session_, price_}` and suffix `{_constant, _user, _item}` such as `item_quality_user`.

Overall, there are four types of additive component, except the error term $$\epsilon$$, in the utility representation:

1. Standalone coefficients $$\lambda, \lambda_i, \lambda_u \in \mathbb{R}$$ representing intercepts and item/user level fixed effects.
2. “Matrix factorization” coefficients $$\theta_u^\top \alpha_i$$, where $$\theta_u,\alpha_i \in \mathbb{R}^L$$ are embedding/latent of users and items, $$L$$ is the latent dimension specified by the researcher.
3. Observable terms $$\zeta_u^\top X^{item}_i$$, where each $$\zeta_u \in \mathbb{R}^{K_{item}}$$ is the user specific coefficients for item observables. This type of component is written as `zeta_user * item_obs` in the utility formula. For sure, one can use coefficients constant among users by simply putting `zeta_constant` in the utility formula.
4.  “Matrix factorization” coefficients of observables written as `gamma_user * beta_item * price_obs`.  This type of component factorizes the coefficient of observables into user and item latents. For example, suppose there are $$K_{price}$$ price observables (i.e., observables varying by both item and session, price is one of them!), for each of price observable $$X^{price}_{is}[k] \in \mathbb{R}$$, a pair of latent $$\gamma_u^k, \beta_i^k \in \mathbb{R}^L$$ is trained to construct the coefficient of the $$k^{th}$$ price observable, where $$L$$ is the latent dimension specified by the researcher. In this case, the utility is  $$\mathcal{U}(u, i, s) = \sum_{k=1}^K (\gamma_u^{k\top} \beta_i^k) X^{price}_{is}[k]$$. One can for sure replace the `price_obs` with any of `{user, item, session}_obs`.

If the researcher wish to treat different part of item observable differently, for example,
$$
\mathcal{U}(u, i, s) = \dots + \zeta_u^\top Y^{item}_i + \omega^\top Z^{item}_i + \dots
$$
where we partition item observables into two parts $$X^{item}_i = [Y^{item}_i, Z^{item}_i]$$, and the coefficient for $$Y^{item}_i$$ is user-specific but the coefficient for the second part is constant across all users.
In this case, the researcher should use separate tensors `item_obs_part1` and `item_obs_part2` while constructing the `ChoiceDataset` (both of them needs to start with `item_`), and use the `utility formula` with `zeta_user * item_obs_part1 + omega_constant * item_obs_part2`.

With the above four cases as building blocks, the researcher can specify all kinds of utility functions.

## Incorporating Observables to the Bayesian Prior

BEMB is a Bayesian factorization model trained by optimizing the evidence lower bound (ELBO). Each parameter (i.e., these with `_item, _user, _constant` suffix.) in the BEMB model carries a prior distribution, which is set to $$\mathcal{N}(\mathbf{0}, \mathbf{I})$$ by default. Beyond this baseline case, the hierarchical nature of BEMB allows the mean of the prior distribution to depend on observables, for example:

$$
\theta_{i} \overset{prior}{\sim} \mathcal{N}(HX^{item}_i, \mathbf{I})
$$

where the prior mean is a linear transformation of the item observable and $$H: \mathbb{R}^{K_{item}} \to \mathbb{R}^L$$.

To enable the observable-to-prior feature, one needs to set `obs2prior_dict['theta_item']=True`.

## Advanced Topics: Additional Modules

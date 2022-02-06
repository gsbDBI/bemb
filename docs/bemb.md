<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
# BEMB Tutorial

## Running BEMB

Running BEMB requires you to (1) build the choice-dataset object and (2) executing the training script.

### ! Advanced Usage: Additional Features

**TODO.**

## Setup the BEMB Model (PyTorch-Lightning Interface)

You will be constructing the `LitBEMBFlex` class, which is a PyTorch-lightning wrapper of the BEMB model implemented in plain PyTorch. The lighting wrapper free researchers from complications such as setting up the training loop and optimizers.

To initialize the `LitBEMBFlex` class, the researcher needs to provide it with the following components. We recommend the research to encompass all arguments in a separate yaml file. Most of these arguments should be self explanatory, Please refer to the doc string of `BEMBFlex.__init__()` for a detailed elaboration.

### Utility Formula

**Note**: for the string parsing to work correctly, please **do** place spaces around `+` and `*`.

The utility formula is a string representing the utility function $$\mathcal{U}(u,i,s)$$ for user $$u$$ to purchase item $$i$$ in session $$s$$. Computing $$\mathcal{U}$$ for all items within the same category $$c$$ and apply the soft-max function of them provides the likelihood of purchasing item $$i$$.

Specifically,

$$
P(i|u,s) = \frac{e^{\mathcal{U}(u, i, s)}}{\sum_{i' \in I_c} e^{\mathcal{U}(u, i', s)}}
$$

The BEMB admits a *linear additive form* of utility formula. For example, the model parses utility formula string `lambda_item + theta_user * alpha_item` into the following representation:

$$
\mathcal{U}(u, i, s)= \lambda_i + \theta_u^\top \alpha_i + \varepsilon \in \mathbb{R}
$$

Overall, there are four types of additive component, except the error term $$\epsilon$$, going into the utility representation:

1. Standalone coefficients $$\lambda, \lambda_i, \lambda_u \in \mathbb{R}$$ representing intercepts and item/user level fixed effects.
2. “Matrix factorization” coefficients $$\theta_u^\top \alpha_i$$, where $$\theta_u,\alpha_i \in \mathbb{R}^L$$ are embedding/latent of users and items, $$L$$ is the latent dimension specified by the researcher.
3. Observable terms $$\zeta_u^\top X^{item}_i$$, where each $$\zeta_u \in \mathbb{R}^{K_{item}}$$ is the user specific coefficients for item observables. This type of component is written as `zeta_user * item_obs` in the utility formula. For sure, one can use coefficients constant among users by simply putting `zeta_constant` in the utility formula.
4.  “Matrix factorization” coefficients of observables written as `gamma_user * beta_item * price_obs`.  This type of component factorizes the coefficient of observables into user and item latents. For example, suppose there are $$K_{price}$$ price observables (i.e., observables varying by both item and session, price is one of them!), for each of price observable $$X^{price}_{is}[k] \in \mathbb{R}$$, a pair of latent $$\gamma_u^k, \beta_i^k \in \mathbb{R}^L$$ is trained to construct the coefficient of the $$k^{th}$$ price observable, where $$L$$ is the latent dimension specified by the researcher. In this case, the utility is  $$\mathcal{U}(u, i, s) = \sum_{k=1}^K (\gamma_u^{k\top} \beta_i^k) X^{price}_{is}[k]$$. One can for sure replace the `price_obs` with any of `{user, item, session}_obs`.

## Incorporating Observables to the Bayesian Prior

BEMB is a Bayesian factorization model trained by optimizing the evidence lower bound (ELBO). Each parameter (i.e., these with `_item, _user, _constant` suffix.) in the BEMB model carries a prior distribution, which is set to $$\mathcal{N}(\mathbf{0}, \mathbf{I})$$ by default. Beyond this baseline case, the hierarchical nature of BEMB allows the mean of the prior distribution to depend on observables, for example:

$$
\theta_{i} \overset{prior}{\sim} \mathcal{N}(HX^{item}_i, \mathbf{I})
$$

where the prior mean is a linear transformation of the item observable and $$H: \mathbb{R}^{K_{item}} \to \mathbb{R}^L$$.

To enable the observable-to-prior feature, one needs to set `obs2prior_dict['theta_item']=True`.
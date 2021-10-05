<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Random Utility Model (RUM) in PyTorch

## Conditional Logit Model

```python
model = ConditionalLogitModel(coef_variation_dict={'price_cost_freq_ovt': 'constant',
                                                   'session_income': 'item',
                                                   'price_ivt': 'item-full',
                                                   'intercept': 'item'},
                              num_param_dict={'price_cost_freq_ovt': 3,
                                              'session_income': 1,
                                              'price_ivt': 1,
                                              'intercept': 1},
                              num_items=4
                              num_users=None)
```

$$
U_{uit} = \beta^0_i + \beta^{1\top} X^{price: (cost, freq, ovt)}_{it} + \beta^2_i X^{session:income}_t + \beta^3_i X_{it}^{price:ivt} + \epsilon_{uit}
$$

The utility for user $u$ to choose item $i$ at time $t$ (i.e., the corresponding session) is modelled as $U_{ijt}$ above. The `deepchoice` allows the option of enforcing coefficient for one item to be zero, the variation of $\beta^3$ is specified as `item-full` which indicates 4 values of $\beta^3$ is learned. In contrast, $\beta^0, \beta^2$ are specified to have variation `item` instead of `item-full`. In this case, the $\beta$ correspond to the frist item (i.e., the baseline item, which is encoded as 0 in the label tensor) is force to be zero.

The model needs to know the dimension of each individual $\beta_i$ (for item-specific coefficients) and $\beta$ (for coefficient constant across items). 

## Nested Logit Model

```python
model = NestedLogitModel(category_to_item=category_to_item,
                         category_coef_variation_dict={},
                         category_num_param_dict={},
                         item_coef_variation_dict={'price_obs': 'constant'},
                         item_num_param_dict={'price_obs': 7},
                         shared_lambda=True)
```

The nested logit model decompoes the utility of choosing item $i$ into the (1) item-specific values and (2) category specify values.  For simplicity, suppose item $i$  belongs to category $k \in \{1, \dots, K\}$: $i \in B_k$. 
$$
U_{uit} = W_{ukt} + Y_{uit}
$$
Where both $W$ and $Y$ are estimated using linear models from as in the conditional logit model.

The log-likelihood for user $u$ to choose item $i$ at time/session $t$ decomposes into the item-level likelihood and category-level likelihood.
$$
\log P(i \mid u, t) &= \log P(i \mid u, t, B_k) + \log P(k \mid u, t) \\
&= \log \left(\frac{\exp(Y_{uit}/\lambda_k)}{\sum_{j \in B_k} \exp(Y_{ujt}/\lambda_k)}\right) + \log \left( \frac{\exp(W_{ukt} + \lambda_k I_{ukt})}{\sum_{\ell=1}^K \exp(W_{u\ell t} + \lambda_\ell I_{u\ell t})}\right)
$$
The **inclusive value** of category $k$, $I_{ukt}$ is defined as $\log \sum_{j \in B_k} \exp(Y_{ujt}/\lambda_k)$, which is the *expecte utility from choosing the best alternative from category $k$*.

The `category_to_item` keyword defines a dictionary of the mapping $k \mapsto B_k$, where keys of `category_to_item`  are integer $k$'s and  `category_to_item[k]`  is a list consisting of IDs of items in $B_k$.

The `{category, item}_coef_variation_dict` provides specification to $W_{ukt}$ and $Y_{uit}$ respectively, `deepchoice` allows for empty category level models by providing an empty dictionary (in this case, $W_{ukt} = \epsilon_{ukt}$) since the inclusive value term $\lambda_k I_{ukt}$ will be used to model the choice over categories. However, by specifying an empty second stage model ($Y_{uit} = \epsilon_{uit}$), the nested logit model reduces to a conditonal logit model of choices over categories. Hence, one should never use the `NestedLogitModel` class with an empty item-level model.

Similar to the conditional logit model, `{category, item}_num_param_dict` specify the dimension (number of observables to be multiplied with the coefficient) of coefficients. The above code initalizes a simple model built upon item-time-specific observables $X_{it} \in \mathbb{R}^7$, 
$$
Y_{uit} = \beta^\top X_{it} + \epsilon_{uit} \\
W_{ukt} = \epsilon_{ukt}
$$
The research may wish to enfoce the *elasiticity* $\lambda_k$ to be constant across categories, setting `shared_lambda=True` enforces $\lambda_k = \lambda\ \forall k \in [K]$.


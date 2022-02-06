<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Random Utility Model (RUM) in PyTorch
This section of the documentation is devoted to cover the theory behind and usage of the two RUM baseline model provided by the package. We also provided Jupyter notebooks tutorials as supplementary materials.

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

The utility for user $u$ to choose item $$i$$ at time $$t$$ (i.e., the corresponding session) is modelled as $$U_{ijt}$$ above. The `deepchoice` allows the option of enforcing coefficient for one item to be zero, the variation of $$\beta^3$$ is specified as `item-full` which indicates 4 values of $$\beta^3$$ is learned. In contrast, $$\beta^0, \beta^2$$ are specified to have variation `item` instead of `item-full`. In this case, the $$\beta$$ correspond to the first item (i.e., the baseline item, which is encoded as 0 in the label tensor) is force to be zero.

The model needs to know the dimension of each individual $$\beta_i$$ (for item-specific coefficients) and $$\beta$$ (for coefficient constant across items).

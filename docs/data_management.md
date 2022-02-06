<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script>
    renderMathInElement(document.body,{delimiters: [
    					{left: "$", right: "$", display: false},
					  {left: "$$", right: "$$", display: true}
]});

</script>
# Data Management
The `torch_choice` and `bemb` packages share the `ChoiceDataset` data structure for managing choice histories.
The `ChoiceDataset` is an instance of the PyTorch dataset object, which allows for easy training with mini-batch sampling.

We provided a Jupyter notebook for this tutorial as well.

## Setup the Choice-Dataset

Example: `bemb_lightning`.

The BEMB model was initially designed for predicting consumers’ purchasing choices from the supermarket purchase dataset, we use the same setup in this tutorial.

We begin with notations and essential factors of the prediction problem. Suppose there are $I$ items under our consideration and the set of **items** $$i \in \{1,2,\dots,I\}$$ can be grouped into $$C$$ **categories** $$c \in \{1,2,\dots,C\}$$, further, let $$I_c$$ denote the collection of items in category $$c$$.

**Example**:

Since we will be using PyTorch to train our model, we represent their identities using integers. Moreover, this document will use lower cases $$i, c$$, etc to index items and categories respectively.

 Let $$B$$ denote the number of purchasing records in the dataset, each $$b \in \{1,2,\dots, B\}$$ record is associated with an **user** $$u \in \{1,2,\dots,U\}$$ and a **session** $$s \in \{1,2,\dots, S\}$$. When there are multiple items bought in the same shopping trip, there will be multiple rows in the dataset with the same $$(u, s)$$.

One canonical example of session $$s$$ is the date of purchase or the shopping trip.

The `ChoiceDataset` data manager is initialized with the following PyTorch tensors:

1. `label` $$\in \{1,2,\dots,I\}^B$$ : the ID of bought item.
2. `user_index` $$\in \{1,2,\dots,U\}^B$$: the ID of the corresponding user (shopper).
3. `session_index` $$\in \{1,2,\dots,S\}^B$$
4. `item_availability` $$\in \{\texttt{True}, \texttt{False}\}^{S\times I}$$  identifies the availability of items in each session, the model will ignore unavailable items while making prediction.
5. `user_obs` $$\in \mathbb{R}^{U\times K_{user}}$$
6. `item_obs` $$\in \mathbb{R}^{I\times K_{item}}$$
7. `price_obs` $$\in \mathbb{R}^{S \times I \times K_{price}}$$

### ! Advanced Usage: Additional Features

**TODO.**

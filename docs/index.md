# Deep Choice: Consumer Choice Modeling with PyTorch
## Introduction
The `deepchoice` is a collection of PyTorch based packages for modeling consumer choices.
With the growing size of choice dataset available (e.g., supermarket purchase records), existing implementation of consumer choice modelling does not easily scale up modern datasets of millions of observations. Our objective is to provides a versatile interface for managing choice dataset, a range of baseline models (the `torch_choice` part), and Bayesian Embedding models for choice modeling (the `bemb` part) that leverage GPU accelerations.

The package leverage GPU acceleration support using PyTorch and easily scale to large dataset of millions of choice records. Beside, we provide easy-to-use PyTorch lightning wrapper of models to free researchers from the hassle from setting up PyTorch optimizers and training loops.

## Example Use Cases
One example use case of this model is to predict shoppers’ purchasing decisions (**Add reference to the shopper paper**).

## Installation
The `deepchoice` release can be installed (1) using `pip` (2) from source code. There are two parts of this project. The `torch_choice` library consisting of data structures and baseline models is standalone, the repository is located [here](). The Bayesian embedding model in `bemb` library relies on components of `torch_choice`, hence we highly recommend users to install `torch_choice` first.

We are still adding features and finalizing the codebase, we highly recommend users to install from source code directly now.

### 1. Install the package using pip
**Not available yet**.

### 2. Install the package from source
1. Clone the repositories of both `torch_choice` and `bemb` to your local machine or server.
2. Install required dependencies (e.g., PyTorch and PyTorch-Lightning).
3. For each of repositories, run `python3 ./setup.py develop` to add the package to your Python environment.
4.  Check installation by running `python3 -c "import torch_choice; print(torch_choice.__version__)"`.
5. Check installation by running `python3 -c "import bemb; print(bemb.__version__)"`.

## User Guide
### Table of Contents
1.[Data Management](./data_management.md)
2. [Tutorial: Random Utility Models](./RUM.md)
3. [Tutorial: Bayesian Embedding (BEMB)](./BEMB.md)
<!-- 3. [API Reference: Dataset](./data.choice_dataset.md)
1. [API Reference: BEMB PyTorch Implementation](./model.bemb_flex_v3.md)
2. [API Reference: BEMB PyTorch Lightning Wrapper](./model.bemb_lightning.md)
3. [API Reference: Bayesian Coefficient](./model.bayesian_coefficient.md)
4. [Jupyter Notebook Example: Conditional Logit Model](../examples/conditional_logit_model_mode_canada.ipynb)
5. [Jupyter Notebook Example: Nested Logit Model](../examples/nested_logit_model_HC.ipynb) -->

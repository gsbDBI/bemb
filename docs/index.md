# Deep Choice: Consumer Choice Modeling with PyTorch
`Deepchoice` package is a PyTorch based package for modeling consumer choices. `deepchoice` provides a versatile interface for managing choice dataset, a range of baseline models, and Bayesian models for choice modeling. One example usage of this model is to predict shoppers’ purchasing decisions (**Add reference to the shopper paper**).

The package leverage GPU acceleration support using PyTorch and easily scale to large dataset of millions of choice records. Beside, we provide easy-to-use PyTorch lightning wrapper of models to free researchers from the hassle from setting up PyTorch optimizers and training loops.

## Installation
The `deepchoice` release can be installed (1) using `pip` package management (2) from source code directly. Since we are still adding features to this project, we highly recommend users to install from source code directly.

### Install the package using pip
**Not available yet**: We are currently finalizing the documentation and will submit `deepchoice` to pip later.

### Install the package from source
1. Clone the repository to your local machine or server.
2. Install required dependencies (e.g., PyTorch and PyTorch-Lightning).
3. Run `python3 ./setup.py develop` to add the package to your Python environment.
4. Check installation by running `python3 -c "import deepchoice; print(deepchoice.__version__)".

## Table of Contents
1. [Tutorial: Random Utility Models](./RUM.md)
2. [Tutorial: Bayesian Embedding (BEMB)](./BEMB.md)
3. [API Reference: Dataset](./data.choice_dataset.md)
4. [API Reference: BEMB PyTorch Implementation](./model.bemb_flex_v3.md)
5. [API Reference: BEMB PyTorch Lightning Wrapper](./model.bemb_lightning.md)
6. [API Reference: Bayesian Coefficient](./model.bayesian_coefficient.md)
7. [Jupyter Notebook Example: Conditional Logit Model](../examples/conditional_logit_model_mode_canada.ipynb)
8. [Jupyter Notebook Example: Nested Logit Model](../examples/nested_logit_model_HC.ipynb)

# Consumer Choice Modeling with PyTorch
This package is a [PyTorch](https://pytorch.org)-based package for, but not limited to, modeling consumer choice.
With the growing size of choice datasets available, existing implementations of consumer choice modelling does not easily scale up modern datasets of millions of records.
Our objective is to provides a versatile interface for managing choice dataset, a range of baseline models (the `torch_choice` part), and a Bayesian Embedding (i.e., BEMB) models for choice modeling that handle large-scale consumer choice datasets in modern research projects.

## Main Components and Features
1. The package includes a data management tool based on `PyTorch`'s dataset called `ChoiceDataset`. Our dataset implementation allows users to easily move data between CPU and GPU.
2. The package provides a conditional logit model for consumer choice modelling.
3. The package provides a nested logit model for consumer choice modelling.
4. The package provides a Bayesian Embedding (also known as probabilistic matrix factorization) model that builds latents for customers and items.
5. The package leverage GPU acceleration using PyTorch and easily scale to large dataset of millions of choice records. All models are trained using state-of-the-art optimizers by in PyTorch. These optimization algorithms are tested to be scalable by modern machine learning practitioners. However, you can rest assure that the package runs flawlessly when no GPU is used as well.
6. For those without much experience in model PyTorch development, setting up optimizers and training loops can be frustrating. We provide easy-to-use [PyTorch lightning](https://www.pytorchlightning.ai) wrapper of models to free researchers from the hassle from setting up PyTorch optimizers and training loops.

## Installation
There are two parts of this project: the `torch_choice` library consisting of data management modules, logit and nested-logit models for consumer choice modelling. The `torch_choice` package offers based

For researchers wish to use the Bayesian Embedding (BEMB) model, they need to install an additional `bemb` package, which was built on the top of `torch_choice`.
We highly recommend users to install `torch_choice` first to have a taste of the type of research question we are trying to solve.

**Note** Since this project is still on its pre-release stage and subject to changes, we have not uploaded our packages to PIP or CONDA. Researchers need to install these packages from Github source code.

To install `torch_choice` and `bemb` (optional) from source:
1. Clone the repositories of both `torch_choice` and `bemb` to your local machine.
2. Install required dependencies (e.g., PyTorch and PyTorch-Lightning).
3. For each of repositories, run `python3 ./setup.py develop` to add the package to your Python environment.
4. Check installation by running `python3 -c "import torch_choice; print(torch_choice.__version__)"`.
5. Check installation by running `python3 -c "import bemb; print(bemb.__version__)"`.


## Usage Example
```python
```

## Research Projects using this Package.
One example use case of this model is to predict shoppers’ purchasing decisions (**Add reference to the shopper paper**).


## User Guide
The development team offers several Jupyter notebook-based and Markdown tutorials covering various aspects of the package.
### Table of Contents
1. [Data Management](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/data_management.ipynb)
2. [Random Utility Model (RUM) 1: Conditional Logit Model](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/conditional_logit_model_mode_canada.ipynb)
3. [Random Utility Model (RUM) 2: Nested Logit Model](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/nested_logit_model_house_cooling.ipynb)
4. [Bayesian Embedding Model (BEMB)](./bemb.md)

## Compatibility Check List
We have tested the tutorials using the following environments, please let us know if there is any issue with our packages on other systems.

| Tutorial | Platform Versions    | CPU | GPU | Device Tested |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| [Data Management](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/data_management.ipynb) | MacOS 12.2 Python 3.9.7 PyTorch 1.10.0 | M1 Max | N/A | cpu |
| [Data Management](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/data_management.ipynb) | Ubuntu 20.04 Python 3.8.10 PyTorch 1.10.1 CUDA 11.3 | 11700F | RTX3090 | cpu and cuda |
| [Conditional Logit Model](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/conditional_logit_model_mode_canada.ipynb) | MacOS 12.2 Python 3.9 PyTorch 1.10.0 | M1 Max | N/A | cpu |
| [Conditional Logit Model](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/conditional_logit_model_mode_canada.ipynb) | Ubuntu 20.04 Python 3.8.10 PyTorch 1.10.1 CUDA 11.3 | 11700F | RTX3090 | cpu and cuda |
| [Nested Logit Model](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/nested_logit_model_house_cooling.ipynb) | MacOS 12.2 Python 3.9.7 PyTorch 1.10.0 | M1 Max | N/A | cpu |
| [Nested Logit Model](https://github.com/gsbDBI/torch-choice/blob/main/tutorials/nested_logit_model_house_cooling.ipynb)| Ubuntu 20.04 Python 3.8.10 PyTorch 1.10.1 CUDA 11.3 | 11700F | RTX3090 | cpu and cuda |
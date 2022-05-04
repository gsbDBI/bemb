# Welcome to torch-choice's documentation!

Authors: Tianyu Du and Ayush Kanodia; PI: Susan Athey; Contact: tianyudu@stanford.edu

This package is a [PyTorch](https://pytorch.org>) based package for, but not limited to, modeling consumer choice.

With the growing size of choice datasets available, existing implementations of consumer choice modelling does not easily scale up modern datasets of millions of records.

Our objective is to provide a versatile interface for managing choice dataset, a range of baseline models (the `torch_choice` part), and a Bayesian Embedding (i.e., BEMB) models for choice modeling that handle large-scale consumer choice datasets in modern research projects.

1. The package includes a data management tool based on `PyTorch`'s dataset called `ChoiceDataset`. Our dataset implementation allows users to easily move data between CPU and GPU. Unlike traditional long or wide formats, the `ChoiceDataset` offers a memory-efficient way to manage observables.

2. The package provides a (1) conditional logit model for consumer choice modeling, (2) a nested logit model for consumer choice modeling, and (3) a Bayesian Embedding (also known as probabilistic matrix factorization) model that builds latents for customers and items.

3. The package leverage GPU acceleration using PyTorch and easily scale to large dataset of millions of choice records. All models are trained using state-of-the-art optimizers by in PyTorch. These optimization algorithms are tested to be scalable by modern machine learning practitioners. However, you can rest assure that the package runs flawlessly when no GPU is used as well.

4. For those without much experience in model PyTorch development, setting up optimizers and training loops can be frustrating. We provide easy-to-use [PyTorch lightning](https://www.pytorchlightning.ai) wrapper of models to free researchers from the hassle from setting up PyTorch optimizers and training loops.


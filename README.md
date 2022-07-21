# Bayesian Embedding (BEMB)

> Authors: Tianyu Du and Ayush Kanodia; PI: Susan Athey; Contact: tianyudu@stanford.edu

BEMB is a flexible, fast Bayesian embedding model for modelling choice problems. The `bemb` package is built upon the [`torch_choice`](https://gsbdbi.github.io/torch-choice/) library.

The full documentation website for BEMB is [https://gsbdbi.github.io/bemb/](https://gsbdbi.github.io/bemb/).

## Installation
1. Install `torch-choice` following steps [here](https://gsbdbi.github.io/torch-choice/).
2. Run the following script to install it.
```bash
# Clone the repository to your local machine or server for tutorials.
git clone "git@github.com:gsbDBI/bemb.git"
# Install required dependencies.
pip3 install -r requirements.txt
# Install bemb from the Pip.
pip3 install bemb
# Check installation.
python3 -c 'import torch_choice; print(torch_choice.__version__)'
```

## Example Usage of BEMB
[Here](https://gsbdbi.github.io/bemb/bemb_obs2prior_simulation/) is a simulation exercise of using `bemb`.
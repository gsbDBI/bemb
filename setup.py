import pathlib
import setuptools
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="bemb",
    version="0.1.7",
    description="A Pytorch Backend Library for Choice Modelling with Bayesian Matrix Factorization",
    long_description=README,
    long_description_content_type="text/markdown",
    url="",
    author="Tianyu Du",
    author_email="tianyudu@stanford.edu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    # install_requires=["torch"],
    # entry_points={
    #     "console_scripts": [
    #         "realpython=.__main__:main",
    #     ]
    # },
)
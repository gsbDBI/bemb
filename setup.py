import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="deepchoice",
    version="0.0.1",
    description="A Pytorch Backend Library for Choice Modelling",
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
    packages=["deepchoice"],
    include_package_data=True,
    # install_requires=["torch"],
    # entry_points={
    #     "console_scripts": [
    #         "realpython=.__main__:main",
    #     ]
    # },
)